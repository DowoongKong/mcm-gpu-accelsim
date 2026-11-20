#include "gpu_tlb.h"
#include <cstdlib>   // getenv, abort (set-hash option)
// ACCELSIM_DIAG flag (gpu-sim.cc): enables the diagnostic-only rows below.
extern const bool g_accelsim_diag;
#include <cassert>
#include <cstdio>
#include <algorithm>

// Hot-path consistency checks, compiled in only with -DXLAT_DEBUG_CHECKS.
#ifdef XLAT_DEBUG_CHECKS
#define XLAT_HOT_ASSERT(x) assert(x)
#else
#define XLAT_HOT_ASSERT(x) ((void)0)
#endif

// Outcome bucket labels for the translation latency breakdown rows.
const char *xlat_bucket_name(unsigned b) {
  switch (b) {
    case XLAT_B_L1_HIT:   return "L1_HIT";
    case XLAT_B_L1_MERGE: return "L1_MERGE";
    case XLAT_B_L2_HIT:   return "L2_HIT";
    case XLAT_B_L2_MERGE: return "L2_MERGE";
    case XLAT_B_PTW:      return "PTW";
    default:              return "UNKNOWN";
  }
}

// l1_tlb_mshr 

l1_tlb_mshr::l1_tlb_mshr(unsigned num_entries, unsigned max_merge)
    : m_num_entries(num_entries), m_max_merge(max_merge) {}

bool l1_tlb_mshr::full(uint64_t vpn) const {
  auto it = m_entries.find(vpn);
  if (it != m_entries.end())
    return it->second.waiters.size() >= m_max_merge;  // existing VPN: capped by max_merge
  return m_entries.size() >= m_num_entries;          // new VPN: table capacity
}

bool l1_tlb_mshr::probe(uint64_t vpn) const {
  return m_entries.count(vpn) != 0;
}

void l1_tlb_mshr::add(uint64_t vpn, unsigned warp_id) {
  m_mutations++;  // enables merges for same-vpn waiters
  XLAT_HOT_ASSERT(!full(vpn));  // one extra hash lookup per add
  m_entries[vpn].waiters.push_back(warp_id);
  // ready stays false until mark_ready() is called
}

void l1_tlb_mshr::mark_ready(uint64_t vpn) {
  auto it = m_entries.find(vpn);
  if (it == m_entries.end()) return;
  if (!it->second.ready) {
    it->second.ready = true;
    m_ready_list.push_back(vpn);
  }
}

bool l1_tlb_mshr::access_ready() const {
  return !m_ready_list.empty();
}

l1_tlb_mshr::l1_fill_token l1_tlb_mshr::next_access() {
  m_mutations++;  // frees a table slot for MSHR-full waiters
  assert(!m_ready_list.empty());
  uint64_t vpn = m_ready_list.front();
  m_ready_list.erase(m_ready_list.begin());
  l1_fill_token tok; tok.vpn = vpn;
  auto it = m_entries.find(vpn);
  if (it != m_entries.end()) { tok.waiters = it->second.waiters; m_entries.erase(it); }
  return tok;   // entry erased here; the fill path never calls evict()
}

void l1_tlb_mshr::evict(uint64_t vpn) {
  m_mutations++;  // frees a table slot for MSHR-full waiters
  m_entries.erase(vpn);
}


//  gpu_l1_tlb 

gpu_l1_tlb::gpu_l1_tlb(unsigned num_sets, unsigned assoc,
                       unsigned mshr_entries, unsigned mshr_max_merge, 
                       unsigned lookup_latency, unsigned sm_id,
                       unsigned page_shift)
    : m_num_sets(num_sets),
      m_assoc(assoc),
      m_sm_id(sm_id),
      m_page_shift(page_shift),
      m_lookup_latency(lookup_latency),
      m_mshr(mshr_entries, mshr_max_merge),
      m_hits(0), m_misses(0), m_mshr_hits(0) {
  if (lookup_latency == 0) {  // the lookup pipe needs at least one stage
    fprintf(stderr, "\nFATAL: L1 TLB (%s) lookup latency 0; use 1 or more.\n",
            page_kind_tag(page_shift));
    fflush(stderr);
    abort();
  }
  m_sets.resize(num_sets, std::vector<l1_tlb_entry_t>(assoc, {false, 0, {0}, 0}));
  m_lookup_pipe.assign(m_lookup_latency, {});
}

int gpu_l1_tlb::find_way(unsigned sidx, uint64_t vpn) const {
  for (unsigned w = 0; w < m_assoc; w++)
    if (m_sets[sidx][w].valid && m_sets[sidx][w].vpn == vpn)
      return (int)w;
  return -1;
}

int gpu_l1_tlb::lru_victim(unsigned sidx) const {
  int victim = 0;
  for (unsigned w = 1; w < m_assoc; w++)
    if (m_sets[sidx][w].lru_time < m_sets[sidx][victim].lru_time)
      victim = (int)w;
  return victim;
}

gpu_l1_tlb::result_t gpu_l1_tlb::access(uint64_t va, unsigned warp_id,
                                        gpu_pte_t &out_pte, uint64_t cur) {
  uint64_t vpn_val = vpn(va);
  unsigned sidx    = set_idx(vpn_val);
  uint64_t rid     = req_key(warp_id, vpn_val);

  // Latency breakdown: t_issue is set only in (2b)/(3), where a first call always lands.

  // (1) Already resolved: HIT if the page is in the array, else use the stored verdict.
  auto rit = m_resolved.find(rid);
  if (rit != m_resolved.end()) {
    int way = find_way(sidx, vpn_val);
    if (way >= 0) {
      m_sets[sidx][way].lru_time = cur;
      out_pte = m_sets[sidx][way].pte;
      // Count a hit only for a HIT verdict; misses and merges were counted in resolve_lookup().
      if (rit->second.r == HIT) m_hits++;
      m_resolved.erase(rit);
      // Latency breakdown: t_done (the PTE reached the warp).
      auto xr = m_xlat_rec.find(rid);
      if (xr != m_xlat_rec.end()) {
        xr->second.t_done = cur;
        xlat_close(rid, xr->second, cur);  // folds and erases
      }
      return HIT;
    }
    // Evicted before the replay but the PTE was delivered: use it, so the warp cannot starve.
    if (rit->second.has_pte) {
      out_pte = rit->second.pte;
      if (rit->second.r == HIT) m_hits++;  // same counting rule as the array path
      m_resolved.erase(rit);
      m_fwd_delivered++;
      auto xr = m_xlat_rec.find(rid);
      if (xr != m_xlat_rec.end()) {
        xr->second.t_done = cur;
        xlat_close(rid, xr->second, cur);  // folds and erases
      }
      return HIT;
    }
    // No PTE, but the VPN is in flight: wait on that fill. Never HIT here: out_pte is unset.
    if (m_mshr.probe(vpn_val)) {
      rit->second.r = MSHR_HIT;
      return MSHR_HIT;
    }
    // Stale verdict (MSHR entry evicted without a fill): drop it and start a new lookup.
    m_resolved.erase(rit);
  }
  // (2) Lookup in flight: wait; a re-poll takes no port and does not restart the timer.
  if (m_inflight_lookup.count(rid)) return LOOKUP_PENDING;
  // (2a) -gpgpu_l1d_vipt: a hit overlaps the L1D access (no cycles); a miss pays the lookup.
  if (m_hit_overlap) {
    const int way = find_way(sidx, vpn_val);
    if (way >= 0 && take_lookup_slot(cur)) {
      m_sets[sidx][way].lru_time = cur;
      out_pte = m_sets[sidx][way].pte;
      m_hits++;
      // Translation latency breakdown: a zero-latency L1_HIT sample, so it still counts.
      xlat_req_rec_t rec;
      rec.t_issue = rec.t_l1_inject_first = rec.t_l1_inject = cur;
      rec.t_l1_resolve = rec.t_fill = rec.t_done = cur;
      rec.bucket = XLAT_B_L1_HIT;
      xlat_fold(m_xlat_acc[XLAT_B_L1_HIT], rec);
      return HIT;
    }
  }
  // (2b) All lookup ports busy: nothing is held, the caller retries.
  if (m_n_ports && m_inflight_lookup.size() >= m_n_ports) {
    m_port_busy_stalls++;
    // Latency breakdown: t_issue starts here; l1_port_polls counts polls, not cycles.
    xlat_req_rec_t &rec = m_xlat_rec[rid];
    if (rec.t_issue == 0) rec.t_issue = cur;
    rec.l1_port_polls++;
    return PORT_BUSY;
  }
  // (3) New request: enter the pipe now, or wait for a lookup slot in m_inject_q.
  m_inflight_lookup.insert(rid);
  if (m_inflight_lookup.size() > m_concurrent_max)
    m_concurrent_max = (unsigned)m_inflight_lookup.size();
  xlat_req_rec_t &rec = m_xlat_rec[rid];
  if (rec.t_issue == 0) rec.t_issue = cur;
  const lookup_slot s{true, rid, vpn_val, warp_id};
  if (take_lookup_slot(cur)) {
    start_lookup(s, cur);
  } else {
    m_inject_q.push_back(s);
    m_inject_queued++;
    if (m_inject_q.size() > m_inject_q_max) m_inject_q_max = (unsigned)m_inject_q.size();
  }
  return LOOKUP_PENDING;
}

// Take one of this cycle's lookup slots; requests waiting in m_inject_q go first.
bool gpu_l1_tlb::take_lookup_slot(uint64_t cur) {
  if (m_lookups_per_cycle == 0) return true;
  if (cur != m_inject_cycle) { m_inject_cycle = cur; m_inject_count = 0; }
  if (!m_inject_q.empty() || m_inject_count >= m_lookups_per_cycle) return false;
  m_inject_count++;
  return true;
}

// Puts a request at the pipe tail; it reaches the head lookup_latency cycles later.
void gpu_l1_tlb::start_lookup(const lookup_slot &s, uint64_t cur) {
  pipe_stage(m_lookup_latency - 1).push_back(s);  // ring tail
  m_pipe_live++;
  // Latency breakdown: t_l1_inject; a restart keeps t_l1_inject_first, so s0_redo holds the redo.
  xlat_req_rec_t &rec = m_xlat_rec[s.rid];
  if (rec.t_l1_inject_first == 0) rec.t_l1_inject_first = cur;
  rec.t_l1_inject = cur;
  rec.t_l1_resolve = 0;   // this epoch has not resolved yet
  rec.t_fill = 0;
}

void gpu_l1_tlb::fill(uint64_t vpn, const gpu_pte_t &pte,
                      uint64_t current_cycle, const xlat_down_t *down) {
  m_fill_epoch++;  // array contents changed; MSHR-full waiters may now hit
  unsigned sidx  = set_idx(vpn);
  int      way   = find_way(sidx, vpn);

  if (way < 0) {
    // Evict LRU
    way = lru_victim(sidx);
  }

  l1_tlb_entry_t &entry = m_sets[sidx][way];
  entry.valid    = true;
  entry.vpn      = vpn;
  entry.pte      = pte;
  entry.lru_time = current_cycle;

  // Stamp t_fill on each waiter before mark_ready(): the ldst pops the waiters this cycle.
  const std::vector<unsigned> *waiters = m_mshr.waiters_of(vpn);
  if (waiters) {
    for (unsigned w : *waiters) {
      // Deliver the PTE to the waiter, so it survives an eviction before the replay.
      auto rv = m_resolved.find(req_key(w, vpn));
      if (rv != m_resolved.end()) { rv->second.pte = pte; rv->second.has_pte = true; }
      auto r = m_xlat_rec.find(req_key(w, vpn));
      if (r == m_xlat_rec.end()) continue;
      if (r->second.t_fill == 0) r->second.t_fill = current_cycle;
      // `down` goes only to the request that sent the L2 miss (merged waiters would go negative).
      if (down && r->second.bucket == XLAT_B_PTW && !r->second.has_down) {
        r->second.down = *down;
        r->second.has_down = 1;
      }
    }
  }

  // ldst_unit::cycle_b() drains the ready entry and replays the waiting warps.
  m_mshr.mark_ready(vpn);
}

void gpu_l1_tlb::cycle(uint64_t cur) {
  // Latency breakdown: sample N (in-flight records) once per cycle.
  m_xlat_cycles++;
  const unsigned live = (unsigned)m_xlat_rec.size();
  m_xlat_inflight_sum += live;
  if (live > m_xlat_inflight_max) m_xlat_inflight_max = live;
  // Nothing in the ring and no waiters: all below is a no-op (the head need not advance).
  if (m_pipe_live == 0 && m_mshr_wait_q.empty() && m_inject_q.empty()) {
#ifdef P4_CHECK
    for (const auto &st : m_lookup_pipe)
      assert(st.empty() && "P4_CHECK: pipe_live 0 but a stage holds a slot");
#endif
    return;
  }

  std::vector<mshr_wait_ent> this_cycle_blocked;  // see the ORDER note below
  // Success bookkeeping shared by the pipe head and the MSHR-full wait queue.
  auto on_resolved = [&](const lookup_slot &head) {
    m_inflight_lookup.erase(head.rid);
    auto it = m_resv_fail_start.find(head.rid);
    if (it != m_resv_fail_start.end()) {
      m_resv_fail_cycles += (cur - it->second);
      // Latency breakdown: MSHR-full blocked time inside s2_lookup.
      auto r = m_xlat_rec.find(head.rid);
      if (r != m_xlat_rec.end())
        r->second.l1_resv_fail += (uint32_t)(cur - it->second);
      m_resv_fail_start.erase(it);
    }
  };
  for (lookup_slot &head : pipe_stage(0)) {   // ring head
    if (!head.valid) continue;
    bool in_flight = false;
    bool resolved = resolve_lookup(head, cur, &in_flight);
    if (resolved) {
      on_resolved(head);
    } else {
      // MSHR full: park on the wait queue; in_flight gives the fail type (merge cap if true).
      this_cycle_blocked.push_back(
          {head, cur, cur, in_flight});
      m_resv_fail_start.emplace(head.rid, cur);
    }
  }
  // Retry order: FIFO appends this cycle's group (oldest first); otherwise it is prepended in
  // its own order (newest group first).
  if (m_mshr_wait_fifo) {
    for (const auto &b : this_cycle_blocked) m_mshr_wait_q.push_back(b);
  } else {
    for (auto it = this_cycle_blocked.rbegin(); it != this_cycle_blocked.rend(); ++it)
      m_mshr_wait_q.push_front(*it);
  }
  // Advance the ring: the drained head becomes the empty new tail.
  m_pipe_live -= (unsigned)pipe_stage(0).size();
  pipe_stage(0).clear();
  m_pipe_head = (m_pipe_head + 1) % m_lookup_latency;
  // Requests waiting for a lookup slot start now, oldest first.
  if (!m_inject_q.empty()) {
    m_inject_cycle = cur;
    m_inject_count = 0;
    while (!m_inject_q.empty() && m_inject_count < m_lookups_per_cycle) {
      start_lookup(m_inject_q.front(), cur);
      m_inject_q.pop_front();
      m_inject_count++;
    }
  }
#ifdef P4_CHECK
  {
    unsigned n = 0;
    for (const auto &st : m_lookup_pipe) n += (unsigned)st.size();
    assert(n == m_pipe_live && "P4_CHECK: pipe_live drift");
  }
#endif
  // Retry MSHR-full waiters (no new lookup latency) only when the fill or MSHR epoch moved;
  // otherwise all would fail again. Skipped cycles are charged to their fail type.
  if (!m_mshr_wait_q.empty()) {
    const uint64_t epoch = m_fill_epoch + m_mshr.mutations();
    if (epoch != m_last_wake_epoch) {
      m_last_wake_epoch = epoch;
      std::deque<mshr_wait_ent> keep;
      for (mshr_wait_ent &w : m_mshr_wait_q) {
        if (w.last_attempt == cur) { keep.push_back(w); continue; }  // enqueued this cycle
        // cycles strictly between attempts would all have failed with w's type
        const uint64_t skipped = cur - w.last_attempt - 1;
        if (skipped) {
          if (w.merge_cap) m_mshr_merge_full += skipped;
          else             m_mshr_entries_full += skipped;
        }
        bool in_flight = false;
        if (resolve_lookup(w.s, cur, &in_flight)) {
          on_resolved(w.s);
        } else {
          w.last_attempt = cur;
          w.merge_cap = in_flight;   // == m_mshr.probe(w.s.vpn); state unchanged, no re-lookup
          keep.push_back(w);
        }
      }
      m_mshr_wait_q.swap(keep);
    }
  }
}


bool gpu_l1_tlb::l2_req_front(uint64_t &vpn_out) const {
  if (m_l2_req_queue.empty()) return false;
  vpn_out = m_l2_req_queue.front().vpn;
  return true;
}
void gpu_l1_tlb::l2_req_pop() { m_l2_req_queue.pop_front(); }

bool gpu_l1_tlb::pop_hit_wakeup(unsigned &warp_id) {
  if (m_hit_wakeup.empty()) return false;
  warp_id = m_hit_wakeup.front();   // resolve order: the first hit wakes first
  m_hit_wakeup.pop_front();
  return true;
}

// Resolve a lookup at the pipe head (or an MSHR-full retry). False only when the MSHR is full.
bool gpu_l1_tlb::resolve_lookup(const lookup_slot &s, uint64_t cur, bool *in_flight_out) {
  unsigned sidx = set_idx(s.vpn);
  // Latency breakdown: t_l1_resolve is stamped only on a true return (left the lookup stage).
  int way = find_way(sidx, s.vpn);
  if (way >= 0) {                              // array hit
    m_sets[sidx][way].lru_time = cur;
    m_resolved[s.rid] = {HIT, true, m_sets[sidx][way].pte};  // m_hits: access() block (1); PTE delivered now
    m_resolve_epoch++;                         // translation-scan memo: not covered by fill/MSHR epochs
    m_hit_wakeup.push_back(s.warp_id);      // wake parked warp (no MSHR entry)
    auto xr = m_xlat_rec.find(s.rid);
    if (xr != m_xlat_rec.end()) {
      xr->second.t_l1_resolve = cur;
      // Already in the array: t_fill = t_l1_resolve.
      xr->second.t_fill  = cur;
      xr->second.bucket = XLAT_B_L1_HIT;
    }
    return true;
  }
  // probe(vpn) and full(vpn) in one lookup; nothing changes the MSHR in between.
  bool in_flight, mshr_full;
  m_mshr.peek(s.vpn, in_flight, mshr_full);
  if (in_flight) {                              // VPN already in flight
    if (mshr_full) {                            // entry merge cap (waiters >= max_merge)
      m_mshr_merge_full++;                      // park; retry until a slot frees / fill lands
      if (in_flight_out) *in_flight_out = true;
      return false;
    }
    m_mshr.add(s.vpn, s.warp_id);
    m_resolved[s.rid] = {MSHR_HIT, false, gpu_pte_t{}}; m_mshr_hits++;  // PTE arrives with the fill
    auto xr = m_xlat_rec.find(s.rid);
    if (xr != m_xlat_rec.end()) {
      xr->second.t_l1_resolve = cur;
      xr->second.bucket = XLAT_B_L1_MERGE;
    }
    return true;
  }
  if (mshr_full) {                               // table full (new VPN, no free entry)
    m_mshr_entries_full++;
    if (in_flight_out) *in_flight_out = false;
    return false;
  }
  m_mshr.add(s.vpn, s.warp_id);                 // add at miss time (capacity-gated)
  {
    auto xr = m_xlat_rec.find(s.rid);
    if (xr != m_xlat_rec.end()) {
      xr->second.t_l1_resolve = cur;
      xr->second.bucket = XLAT_B_PTW;
    }
  }
  m_l2_req_queue.push_back({s.vpn, s.warp_id, cur, 0, 0, false, 0, 0});  // hand off to L2 fifo, poll now
  m_l2_req_enq++;
  if (m_l2_req_queue.size() > m_l2_req_q_max)
    m_l2_req_q_max = (unsigned)m_l2_req_queue.size();
  m_misses++;
  // Later polls of this request read MSHR_HIT until the fill.
  m_resolved[s.rid] = {MSHR_HIT, false, gpu_pte_t{}};  // PTE arrives with the fill
  return true;
}

// Latency breakdown: fold a finished request into its bucket; an incomplete record is dropped.
void gpu_l1_tlb::xlat_close(uint64_t rid, xlat_req_rec_t &rec, uint64_t cur) {
  (void)cur;
  if (xlat_rec_complete(rec)) {
    // Refine the PTW bucket by the L2 answer; without `down` it stays (smid unsplit).
    if (rec.bucket == XLAT_B_PTW && rec.has_down) {
      rec.bucket = (rec.down.l2_class == XLAT_L2C_HIT)   ? XLAT_B_L2_HIT
                 : (rec.down.l2_class == XLAT_L2C_MERGE) ? XLAT_B_L2_MERGE
                                                         : XLAT_B_PTW;
    }
    unsigned o = rec.bucket < XLAT_B_NUM ? rec.bucket : XLAT_B_L1_HIT;
    xlat_fold(m_xlat_acc[o], rec);
  } else {
    m_xlat_dropped++;
  }
  m_xlat_rec.erase(rid);
}

// GPU-wide [XLATSUM]: each SM's print_stats() folds in here; xlatsum_gpu_print() prints and resets.
static xlat_sum_t g_xlatsum[3][XLAT_B_NUM];  // kind: 0=4k 1=64k 2=2m
static int xlatsum_kind_idx(const char* kind) {
  if (kind[0]=='4') return 0;
  if (kind[0]=='6') return 1;
  return 2;
}
void xlatsum_gpu_fold(const char* kind, unsigned outcome, const xlat_accum_t &a) {
  g_xlatsum[xlatsum_kind_idx(kind)][outcome].add(a);
}
void xlatsum_gpu_print(FILE *f) {
  static const char* kinds[3] = {"4k", "64k", "2m"};
  for (int k = 0; k < 3; k++) {
    xlat_sum_t all;
    for (unsigned o = 0; o < XLAT_B_NUM; o++) {
      g_xlatsum[k][o].print(f, kinds[k], "GPU", xlat_bucket_name(o));
      // GPU-wide s6_pwqwait/s7_walk histograms ([XLATHIST] rows).
      g_xlatsum[k][o].print_hist(f, kinds[k], "GPU", xlat_bucket_name(o));
      all.add(g_xlatsum[k][o]);
      g_xlatsum[k][o] = xlat_sum_t();
    }
    all.print(f, kinds[k], "GPU", "ALL");
  }
}

void gpu_l1_tlb::print_stats(FILE *f) const {
  unsigned long long total = m_hits + m_misses;
  // Row tag is the page size. hit_rate = hits / (hits + misses); merges are in neither term.
  fprintf(f, "[L1_TLB_%s_STAT] SM=%u hits=%llu misses=%llu mshr_hits=%llu "
             "merge_full=%llu entries_full=%llu resv_fail_cycles=%llu "
             "hit_rate=%.2f%%\n",
          page_kind_tag(m_page_shift),
          m_sm_id, m_hits, m_misses, m_mshr_hits,
          m_mshr_merge_full, m_mshr_entries_full, m_resv_fail_cycles,
          total ? 100.0 * m_hits / total : 0.0);
  // Printed only if the delivered-PTE path was used.
  if (m_fwd_delivered)
    fprintf(f, "[L1_TLB_%s_FWD] SM=%u evicted_before_consume_delivered=%llu\n",
            page_kind_tag(m_page_shift), m_sm_id, m_fwd_delivered);
  // Port and queue limits; the PORT row prints only with a port cap or under ACCELSIM_DIAG.
  if (m_lookups_per_cycle != 0)
    fprintf(f, "[L1_TLB_%s_LOOKUP] SM=%u lookups_per_cycle=%u queued=%llu queue_max=%u\n",
            page_kind_tag(m_page_shift), m_sm_id, m_lookups_per_cycle,
            m_inject_queued, m_inject_q_max);
  if (m_n_ports != 0 || g_accelsim_diag)
    fprintf(f, "[L1_TLB_%s_PORT] SM=%u ports=%u port_busy=%llu concurrent_max=%u l2_req_q_max=%u\n",
            page_kind_tag(m_page_shift), m_sm_id, m_n_ports,
            m_port_busy_stalls, m_concurrent_max, m_l2_req_q_max);
  // [L1_TLB_<kind>_XLATLAT]: latency breakdown per outcome. resid must be 0 (segments tile
  // [t_issue, t_done]); smid is the unsplit middle (L2 + PWQ + PTW + fill).
  for (unsigned o = 0; o < XLAT_B_NUM; o++) {
    const xlat_accum_t &a = m_xlat_acc[o];
    if (a.n == 0) continue;
    const double n = (double)a.n;
    fprintf(f,
            "[L1_TLB_%s_XLATLAT] SM=%u out=%s n=%llu total=%.2f s0_redo=%.2f "
            "s1_admit=%.2f s2_lookup=%.2f smid=%.2f s3_link=%.2f s4_l2look=%.2f "
            "s5_pwqadmit=%.2f rest=%.2f s9_replay=%.2f l1_resv_fail=%.2f "
            "l2_resv_fail=%.2f port_polls=%.3f l2_polls=%.3f l2_reprobe=%.3f "
            "unsplit=%llu resid=%llu\n",
            page_kind_tag(m_page_shift), m_sm_id, xlat_bucket_name(o), a.n,
            a.total / n, a.s0 / n, a.s1 / n, a.s2 / n, a.smid / n, a.s3 / n,
            a.s4 / n, a.s5 / n, a.rest / n, a.s9 / n, a.l1_resv_fail / n,
            a.l2_resv_fail / n, a.l1_port_polls / n, a.l2_port_polls / n,
            a.l2_reprobe / n, a.unsplit, a.residual);
    // Walk row: s6/s7/s8 average over n_walk (requests that walked), not n.
    if (a.n_walk) {
      const double w = (double)a.n_walk;
      fprintf(f,
              "[L1_TLB_%s_XLATWALK] SM=%u out=%s n_walk=%llu s6_pwqwait=%.2f "
              "s7_walk=%.2f s8_deliver=%.2f steps=%.3f remote=%.3f "
              "pwc_hit=%.3f icnt_bp=%.3f\n",
              page_kind_tag(m_page_shift), m_sm_id, xlat_bucket_name(o),
              a.n_walk, a.s6 / w, a.s7 / w, a.s8 / w, a.steps / w,
              a.remote / w, a.pwc_hit / w, a.icnt_bp / w);
      // Per-SM s6/s7 histograms (bin k covers [2^(k-1), 2^k) cycles), only under ACCELSIM_DIAG.
      if (g_accelsim_diag) {
        fprintf(f, "[L1_TLB_%s_XLATHIST] SM=%u out=%s kind=s6_pwqwait n=%llu",
                page_kind_tag(m_page_shift), m_sm_id, xlat_bucket_name(o),
                a.n_walk);
        xlat_hist_print(f, a.hist_s6);
        fprintf(f, "\n[L1_TLB_%s_XLATHIST] SM=%u out=%s kind=s7_walk n=%llu",
                page_kind_tag(m_page_shift), m_sm_id, xlat_bucket_name(o),
                a.n_walk);
        xlat_hist_print(f, a.hist_s7);
        fprintf(f, "\n");
      }
    }
  }
  fprintf(f, "[L1_TLB_%s_XLATLAT] SM=%u dropped=%llu inflight_at_end=%zu\n",
          page_kind_tag(m_page_shift), m_sm_id, m_xlat_dropped,
          m_xlat_rec.size());
  // [XLATSUM]: one ALL row per SM; per-outcome rows are printed GPU-wide by xlatsum_gpu_print().
  {
    xlat_sum_t sm_all;
    for (unsigned o = 0; o < XLAT_B_NUM; o++) {
      sm_all.add(m_xlat_acc[o]);
      xlatsum_gpu_fold(page_kind_tag(m_page_shift), o, m_xlat_acc[o]);
    }
    char head[32]; snprintf(head, sizeof(head), "SM=%u", m_sm_id);
    sm_all.print(f, page_kind_tag(m_page_shift), head, "ALL");
  }
  // Little's Law check (N = X*W; a mismatch means a bad boundary), only under ACCELSIM_DIAG.
  if (g_accelsim_diag) {
    unsigned long long done = 0, wsum = 0;
    for (unsigned o = 0; o < XLAT_B_NUM; o++) {
      done += m_xlat_acc[o].n;
      wsum += m_xlat_acc[o].total;
    }
    const double c = m_xlat_cycles ? (double)m_xlat_cycles : 1.0;
    fprintf(f,
            "[L1_TLB_%s_XLATLITTLE] SM=%u cycles=%llu N_avg=%.4f N_max=%u "
            "X=%.6f W=%.2f XW=%.4f completions=%llu\n",
            page_kind_tag(m_page_shift), m_sm_id, m_xlat_cycles,
            (double)m_xlat_inflight_sum / c, m_xlat_inflight_max,
            (double)done / c, done ? (double)wsum / done : 0.0,
            done ? ((double)done / c) * ((double)wsum / done) : 0.0, done);
  }
}

// gpu_l2_tlb
gpu_l2_tlb::gpu_l2_tlb(unsigned num_sets, unsigned assoc,
                        unsigned mshr_entries, unsigned mshr_max_merge,
                        unsigned lookup_latency, unsigned chiplet_id,
                        unsigned page_shift)
    : m_num_sets(num_sets), m_assoc(assoc), m_chiplet_id(chiplet_id),
      m_page_shift(page_shift), m_mshr_entries(mshr_entries),
      m_mshr_max_merge(mshr_max_merge), m_lookup_latency(lookup_latency),
      m_hits(0), m_misses(0), m_mshr_hits(0) {
  m_sets.resize(num_sets,
                std::vector<l1_tlb_entry_t>(assoc, {false, 0, {0}, 0}));
  m_ep_set.assign(num_sets, 1);  // per-set epochs (see table_full())
  m_set_lookups.assign(num_sets, 0);
  m_set_fills.assign(num_sets, 0);
  m_set_bits = 0;
  while ((1u << m_set_bits) < num_sets) m_set_bits++;
}

void gpu_l2_tlb::set_set_hash(unsigned mode) {
  m_set_hash = mode;
  if (mode != 0 && (m_num_sets < 2 || (m_num_sets & (m_num_sets - 1)) != 0)) {
    fprintf(stderr, "FATAL: -gpgpu_l2_tlb_set_hash needs a power-of-two set count >= 2 (got %u)\n", m_num_sets);
    fflush(stderr);
    abort();
  }
}

bool gpu_l2_tlb::full_verdict(uint64_t vpn_val, bool *merge) const {  // debug check build (P4_CHECK)
  if (find_way(set_idx(vpn_val), vpn_val) >= 0) return false;
  auto it = m_mshr.find(vpn_val);
  if (it != m_mshr.end()) {
    *merge = true;
    return it->second.pending_clusters.size() >= m_mshr_max_merge;
  }
  *merge = false;
  return m_mshr.size() >= m_mshr_entries;
}

int gpu_l2_tlb::find_way(unsigned sidx, uint64_t vpn_val) const {
  for (unsigned w = 0; w < m_assoc; w++)
    if (m_sets[sidx][w].valid && m_sets[sidx][w].vpn == vpn_val)
      return (int)w;
  return -1;
}

int gpu_l2_tlb::lru_victim(unsigned sidx) const {
  int victim = 0;
  for (unsigned w = 1; w < m_assoc; w++)
    if (m_sets[sidx][w].lru_time < m_sets[sidx][victim].lru_time)
      victim = (int)w;
  return victim;
}

gpu_l2_tlb::result_t gpu_l2_tlb::access(uint64_t va, unsigned cluster_id,
                                        gpu_pte_t& out_pte, uint64_t cur,
                                        uint64_t* ready_out,
                                        bool* was_probing_out) {
  uint64_t vpn_val = vpn(va);
  unsigned sidx    = set_idx(vpn_val);
  uint64_t rid     = req_key(cluster_id, vpn_val);

  // (1) Inside this request's OWN probe-latency window (keyed by rid, not vpn).
  auto git = m_lookup_ready.find(rid);
  // Keys of m_lookup_ready == m_probing, so this find also answers is_probing().
  if (was_probing_out) *was_probing_out = (git != m_lookup_ready.end());
  if (git != m_lookup_ready.end()) {
    if (cur < git->second.ready) {
      m_gate_pending++;
      if (ready_out) *ready_out = git->second.ready;  // let the caller sleep until expiry
      return LOOKUP_PENDING;
    }
    m_gate_expire++;
    // resolve() leaves m_lookup_ready alone, so `git` stays valid.
    result_t r = resolve(vpn_val, sidx, cluster_id, out_pte, cur);
    // Track time blocked on MSHR-full, as in gpu_l1_tlb.
    if (r == MSHR_FULL) {
      // Keep the port (still in the lookup stage); record the first blocked cycle once.
      if (!git->second.blocked) {
        m_resv_fail_start.emplace(rid, cur);
        git->second.blocked = true;
      }
      // Re-arm as already expired: the next poll retries without paying the lookup latency again.
      git->second.ready = cur;
    } else {
      m_lookup_ready.erase(git);                      // probe latency elapsed
      m_probing.erase(rid);                           // left the lookup stage
      auto fit = m_resv_fail_start.find(rid);
      if (fit != m_resv_fail_start.end()) {
        m_resv_fail_cycles += (cur - fit->second);
        // Translation latency breakdown: sub-attribution inside s4_l2look.
        auto xs = m_xlat_seg.find(rid);
        if (xs != m_xlat_seg.end())
          xs->second.l2_resv_fail += (uint32_t)(cur - fit->second);
        m_resv_fail_start.erase(fit);
      }
      // Latency breakdown: t_l2_miss = verdict cycle; HIT and MSHR_HIT never enter the PWQ.
      auto xs = m_xlat_seg.find(rid);
      if (xs != m_xlat_seg.end() && xs->second.t_l2_miss == 0) {
        xs->second.t_l2_miss = cur;
        xs->second.l2_class = (r == HIT)      ? XLAT_L2C_HIT
                            : (r == MSHR_HIT) ? XLAT_L2C_MERGE
                                              : XLAT_L2C_MISS;
        if (r != MISS) xs->second.t_pwq_enq = cur;
      }
    }
    return r;
  }
  // (2b) Finite lookup ports: nothing is armed and no resource is held.
  if (m_n_ports && m_probing.size() >= m_n_ports) {
    m_port_busy_stalls++;
    m_xlat_seg[rid].l2_port_polls++;   // latency breakdown: poll COUNT, not cycles
    return PORT_BUSY;
  }
  // (2c) Per-cycle start limit (-gpgpu_l2_tlb_starts_per_cycle); a refused request retries.
  if (m_starts_per_cycle) {
    if (cur != m_start_cycle) { m_start_cycle = cur; m_starts_this_cycle = 0; }
    if (m_starts_this_cycle >= m_starts_per_cycle) {
      m_start_busy++;
      m_xlat_seg[rid].l2_port_polls++;   // latency breakdown: poll COUNT, not cycles
      return PORT_BUSY;
    }
    m_starts_this_cycle++;
  }
  // (3) Brand-new request: take a port and arm the probe-latency gate.
  m_gate_arm_new++;
  m_probing.insert(rid);
  if (m_probing.size() > m_concurrent_max)
    m_concurrent_max = (unsigned)m_probing.size();
  XLAT_HOT_ASSERT(m_resv_fail_start.count(rid) == 0);  // a fresh gate has no blocked start
  m_lookup_ready[rid] = gate_t{cur + m_lookup_latency, false};
  {
    // Latency breakdown: t_l2_arm keeps the first arm; a later arm counts as a re-probe.
    xlat_down_t& seg = m_xlat_seg[rid];
    if (seg.t_l2_arm == 0) seg.t_l2_arm = cur;
    else                   seg.l2_reprobe++;
  }
  return LOOKUP_PENDING;
}

// Latency breakdown: the GMMU accepted the request into the PWQ (t_pwq_enq).
void gpu_l2_tlb::xlat_mark_pwq_enq(unsigned cluster_id, uint64_t vpn_val,
                                   uint64_t cur) {
  auto it = m_xlat_seg.find(req_key(cluster_id, vpn_val));
  if (it != m_xlat_seg.end() && it->second.t_pwq_enq == 0)
    it->second.t_pwq_enq = cur;
}

// Latency breakdown: hand the L2 record to the fill path and drop it.
bool gpu_l2_tlb::xlat_take_seg(unsigned cluster_id, uint64_t vpn_val,
                               xlat_down_t& out) {
  auto it = m_xlat_seg.find(req_key(cluster_id, vpn_val));
  if (it == m_xlat_seg.end()) return false;
  out = it->second;
  m_xlat_seg.erase(it);
  return true;
}

// Runs when the probe gate expires: array first, then the MSHR (merge or new entry).
gpu_l2_tlb::result_t
gpu_l2_tlb::resolve(uint64_t vpn_val, unsigned sidx, unsigned cluster_id,
                    gpu_pte_t& out_pte, uint64_t cur) {
  int way = find_way(sidx, vpn_val);
  if (way >= 0) {                                   // array hit
    m_sets[sidx][way].lru_time = cur;
    out_pte = m_sets[sidx][way].pte;
    m_hits++;
    m_set_lookups[sidx]++;
    return HIT;
  }
  auto mit = m_mshr.find(vpn_val);
  if (mit != m_mshr.end()) {                        // walk in flight -> merge
    if (mit->second.pending_clusters.size() >= m_mshr_max_merge) {
      m_mshr_merge_full++;                          // entry merge cap
      m_last_full_merge = true;                     // fail type for skip synthesis
      return MSHR_FULL;                             // merge cap; re-poll (gate re-armed as expired)
    }
    mit->second.pending_clusters.push_back(cluster_id);
    if (mit->second.pte_ready) {  // merge into a ready entry: wake the pollers as fill() does
      ready_add(cluster_id, vpn_val);
      m_fill_seq++;
    }
    // A merge cannot flip a frozen MSHR_FULL verdict, so no epoch moves.
    m_mshr_hits++;
    m_set_lookups[sidx]++;
    return MSHR_HIT;
  }
  if (m_mshr.size() >= m_mshr_entries) {            // MSHR table full
    m_mshr_entries_full++;                          // table full
    m_last_full_merge = false;                      // fail type for skip synthesis
    return MSHR_FULL;                               // re-poll (gate re-armed as expired)
  }
  m_ep_set[sidx]++;  // a vpn of this set now exists in the MSHR
  m_ep_set_any++;
  l2_mshr_entry_t& e = m_mshr[vpn_val];             // first miss for this vpn
  e.pending_clusters.push_back(cluster_id);
  e.pte_ready = false;
  e.pte.raw   = 0;
  m_misses++;
  m_set_lookups[sidx]++;
  return MISS;
}

void gpu_l2_tlb::fill(uint64_t vpn_val, const gpu_pte_t& pte,
                       uint64_t current_cycle, const xlat_walk_t* walk) {
  m_fill_seq++;    // wake the pollers (see fill_seq())
  // Insert into TLB array with LRU replacement.
  unsigned sidx = set_idx(vpn_val);
  m_ep_set[sidx]++;  // this set's array contents changed
  m_ep_set_any++;
  int      way  = find_way(sidx, vpn_val);
  if (way < 0) way = lru_victim(sidx);
  m_sets[sidx][way] = {true, vpn_val, pte, current_cycle};
  m_set_fills[sidx]++;

  // Mark the MSHR entry ready, if any.
  auto it = m_mshr.find(vpn_val);
  if (it != m_mshr.end()) {
    it->second.pte_ready = true;
    it->second.pte       = pte;
    for (unsigned cl : it->second.pending_clusters) ready_add(cl, vpn_val);  // see ready_for()

    // Latency breakdown: every waiting cluster shared the walk, so each gets its stats.
    if (walk) {
      for (unsigned cl : it->second.pending_clusters) {
        auto xs = m_xlat_seg.find(req_key(cl, vpn_val));
        if (xs == m_xlat_seg.end()) continue;
        xs->second.t_walk_start = walk->t_walk_start;
        xs->second.t_walk_done  = current_cycle;
        xs->second.n_steps      = walk->n_steps;
        xs->second.n_remote     = walk->n_remote;
        xs->second.n_pwc_hit    = walk->n_pwc_hit;
        xs->second.icnt_bp      = walk->icnt_bp;
      }
    }
  }
}

bool gpu_l2_tlb::query_completed(uint64_t vpn_val, unsigned cluster_id,
                                   gpu_pte_t& out_pte) {
  auto it = m_mshr.find(vpn_val);
  if (it == m_mshr.end() || !it->second.pte_ready) return false;

  auto& clusters = it->second.pending_clusters;
  auto  cit = std::find(clusters.begin(), clusters.end(), cluster_id);
  if (cit == clusters.end()) return false;

  out_pte = it->second.pte;
  clusters.erase(cit);
  // ready_for(): the condition is false once no occurrence of cluster_id remains.
  if (std::find(clusters.begin(), clusters.end(), cluster_id) == clusters.end())
    ready_del(cluster_id, vpn_val);
  m_ep_consume++;  // a merge count shrank
  if (clusters.empty()) m_mshr.erase(it);  // all SMs consumed; free entry (table_full() reads the size live)
  return true;
}

bool gpu_l2_tlb::completable(uint64_t vpn_val, unsigned cluster_id) const {  // debug check build (P4_CHECK)
  auto it = m_mshr.find(vpn_val);
  if (it == m_mshr.end() || !it->second.pte_ready) return false;
  const auto& clusters = it->second.pending_clusters;
  return std::find(clusters.begin(), clusters.end(), cluster_id) != clusters.end();
}

void gpu_l2_tlb::remove_from_mshr(uint64_t vpn_val, unsigned cluster_id) {
  auto it = m_mshr.find(vpn_val);
  if (it == m_mshr.end()) return;
  auto& c = it->second.pending_clusters;
  c.erase(std::remove(c.begin(), c.end(), cluster_id), c.end());
  ready_del(cluster_id, vpn_val);  // every occurrence is gone
  m_ep_consume++;  // a merge count shrank
  if (c.empty()) m_mshr.erase(it);  // table_full() reads the size live
}

void gpu_l2_tlb::print_stats(FILE* f) const {
  unsigned long long total = m_hits + m_misses;
  // Same tag and hit_rate rule as gpu_l1_tlb::print_stats().
  fprintf(f,
          "[L2_TLB_%s_STAT] chiplet=%u hits=%llu misses=%llu "
          "mshr_hits=%llu merge_full=%llu entries_full=%llu "
          "resv_fail_cycles=%llu hit_rate=%.2f%%\n",
          page_kind_tag(m_page_shift),
          m_chiplet_id, m_hits, m_misses,
          m_mshr_hits, m_mshr_merge_full, m_mshr_entries_full,
          m_resv_fail_cycles,
          total ? 100.0 * m_hits / total : 0.0);
  // PORT row only with a port cap or under ACCELSIM_DIAG.
  if (m_n_ports != 0 || g_accelsim_diag)
    fprintf(f, "[L2_TLB_%s_PORT] chiplet=%u ports=%u port_busy=%llu concurrent_max=%u\n",
            page_kind_tag(m_page_shift), m_chiplet_id, m_n_ports,
            m_port_busy_stalls, m_concurrent_max);
  if (m_starts_per_cycle != 0)
    fprintf(f, "[L2_TLB_%s_START] chiplet=%u starts_per_cycle=%u start_busy=%llu\n",
            page_kind_tag(m_page_shift), m_chiplet_id, m_starts_per_cycle,
            m_start_busy);
  // Per-set usage rows, only with ACCELSIM_TLB_SETSTAT=1 (default logs stay unchanged).
  static const bool setstat = (getenv("ACCELSIM_TLB_SETSTAT") != NULL);
  if (setstat) {
    unsigned used_l = 0, used_f = 0; unsigned long long lmin = ~0ULL, lmax = 0, fmin = ~0ULL, fmax = 0, ftot = 0;
    for (unsigned s = 0; s < m_num_sets; s++) {
      if (m_set_lookups[s]) used_l++;
      if (m_set_fills[s]) used_f++;
      lmin = m_set_lookups[s] < lmin ? m_set_lookups[s] : lmin; lmax = m_set_lookups[s] > lmax ? m_set_lookups[s] : lmax;
      fmin = m_set_fills[s] < fmin ? m_set_fills[s] : fmin; fmax = m_set_fills[s] > fmax ? m_set_fills[s] : fmax;
      ftot += m_set_fills[s];
    }
    fprintf(f, "[L2_TLB_%s_SETS] chiplet=%u hash=%u sets=%u sets_with_lookups=%u sets_with_fills=%u "
               "lookups_per_set_min=%llu max=%llu fills_per_set_min=%llu max=%llu fills_total=%llu\n",
            page_kind_tag(m_page_shift), m_chiplet_id, m_set_hash, m_num_sets, used_l, used_f,
            lmin, lmax, fmin, fmax, ftot);
    fprintf(f, "[L2_TLB_%s_SETFILLS] chiplet=%u", page_kind_tag(m_page_shift), m_chiplet_id);
    for (unsigned s = 0; s < m_num_sets; s++) fprintf(f, " %llu", m_set_fills[s]);
    fprintf(f, "\n");
  }
}

// Deadlock dump (XLATDBG_DEADLOCK rows): L1 TLB in-flight state. inflight > 0 with an empty
// pipe means requests wait for a lookup slot or on the MSHR-full queue.
void gpu_l1_tlb::xlatdbg_dump(FILE* fp, unsigned sid, const char* tag) const {
  size_t pipe = 0;
  for (const auto& stage : m_lookup_pipe) pipe += stage.size();
  if (pipe == 0 && m_inflight_lookup.empty() && m_resolved.empty() &&
      m_l2_req_queue.empty() && m_mshr.size() == 0)
    return;   // nothing outstanding: stay quiet so the dump stays readable
  fprintf(fp, "XLATDBG_DEADLOCK:   sm=%u tlb=%s pipe=%zu inflight=%zu resolved=%zu "
              "l2_req_q=%zu mshr_entries=%u",
          sid, tag, pipe, m_inflight_lookup.size(), m_resolved.size(),
          m_l2_req_queue.size(), m_mshr.size());
  // Front = oldest request.
  if (!m_l2_req_queue.empty())
    fprintf(fp, " front_vpn=0x%llx front_warp=%u queued_at=%llu",
            (unsigned long long)m_l2_req_queue.front().vpn,
            m_l2_req_queue.front().warp_id,
            (unsigned long long)m_l2_req_queue.front().enqueue_cycle);
  fprintf(fp, "\n");
}

// Deadlock dump: L2 TLB MSHR occupancy and reject counters (merge cap vs table full).
void gpu_l2_tlb::xlatdbg_dump(FILE* fp, unsigned chiplet, const char* tag) const {
  fprintf(fp, "XLATDBG_DEADLOCK: l2tlb chiplet=%u %s mshr=%zu/%u lookup_ready=%zu "
              "hits=%llu misses=%llu mshr_hits=%llu merge_full=%llu entries_full=%llu\n",
          chiplet, tag, m_mshr.size(), m_mshr_entries, m_lookup_ready.size(),
          m_hits, m_misses, m_mshr_hits, m_mshr_merge_full, m_mshr_entries_full);
  // arm_new far above the distinct request count means gates are re-armed (latency paid again).
  fprintf(fp, "XLATDBG_DEADLOCK:   gate: arm_new=%llu pending=%llu expire=%llu\n",
          m_gate_arm_new, m_gate_pending, m_gate_expire);
  // A pte_ready entry that still has waiters is a fill not yet consumed.
  size_t ready_with_waiters = 0, not_ready = 0;
  for (const auto& kv : m_mshr) {
    if (kv.second.pte_ready && !kv.second.pending_clusters.empty()) ready_with_waiters++;
    if (!kv.second.pte_ready) not_ready++;
  }
  fprintf(fp, "XLATDBG_DEADLOCK:   mshr entries: pte_ready_with_waiters=%zu awaiting_walk=%zu\n",
          ready_with_waiters, not_ready);
}

// Deadlock dump: what access() would answer for `vpn` right now, without mutating.
void gpu_l2_tlb::xlatdbg_probe(FILE* fp, uint64_t vpn, unsigned cluster_id,
                               uint64_t cur) const {
  const uint64_t rid = req_key(cluster_id, vpn);
  auto rit = m_lookup_ready.find(rid);
  const char* gate = "no_gate";
  unsigned long long ready_at = 0;
  if (rit != m_lookup_ready.end()) {
    ready_at = (unsigned long long)rit->second.ready;
    gate = (cur < rit->second.ready) ? "LOOKUP_PENDING" : "gate_expired";
  }
  auto mit = m_mshr.find(vpn);
  const char* mshr_state = "absent";
  size_t waiters = 0;
  if (mit != m_mshr.end()) {
    waiters = mit->second.pending_clusters.size();
    mshr_state = mit->second.pte_ready ? "present_pte_ready" : "present_awaiting_walk";
  }
  fprintf(fp, "XLATDBG_DEADLOCK:   probe vpn=0x%llx cluster=%u cur=%llu gate=%s "
              "ready_at=%llu mshr=%s waiters=%zu table=%zu/%u\n",
          (unsigned long long)vpn, cluster_id, (unsigned long long)cur, gate,
          ready_at, mshr_state, waiters, m_mshr.size(), m_mshr_entries);
}
