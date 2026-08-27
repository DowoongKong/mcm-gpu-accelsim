#include "gpu_tlb.h"
#include <cassert>
#include <cstdio>
#include <algorithm>

// 44-2: pure consistency checks on hot paths. Compiled in only with
// -DXLAT_DEBUG_CHECKS; the default build skips them (the checks re-walk the
// page table / re-hash the MSHR and cost real time at high walk rates).
#ifdef XLAT_DEBUG_CHECKS
#define XLAT_HOT_ASSERT(x) assert(x)
#else
#define XLAT_HOT_ASSERT(x) ((void)0)
#endif

// XLATLAT: bucket labels for the per-request breakdown lines.
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
  m_mutations++;  // 36a: enables merges for same-vpn waiters
  XLAT_HOT_ASSERT(!full(vpn));  // 44-2: one extra hash per add as a plain assert
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
  m_mutations++;  // 36a: frees a table slot
  assert(!m_ready_list.empty());
  uint64_t vpn = m_ready_list.front();
  m_ready_list.erase(m_ready_list.begin());
  l1_fill_token tok; tok.vpn = vpn;
  auto it = m_entries.find(vpn);
  if (it != m_entries.end()) { tok.waiters = it->second.waiters; m_entries.erase(it); }
  return tok;   // entry erased here (folds the old evict())
}

void l1_tlb_mshr::evict(uint64_t vpn) {
  m_mutations++;  // 36a: frees a table slot
  m_entries.erase(vpn);
}

std::vector<uint64_t> l1_tlb_mshr::get_pending_vpns() const {
  std::vector<uint64_t> result;
  for (auto &kv : m_entries)
    if (!kv.second.ready)
      result.push_back(kv.first);
  return result;
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

  // XLATLAT: t0 is recorded lazily, in blocks (2b) and (3) only. The first call
  // for a (warp, vpn) cannot reach blocks (1) or (2) -- nothing is resolved and
  // nothing is in the pipe yet -- so it always lands in (2b) or (3), and t0 is
  // captured with no loss. Keeping the map out of this prologue matters: a parked
  // warp re-polls every page every replay and almost always exits through block
  // (1)'s in-flight branch or block (2), which now touch no map at all.

  // (1) Already resolved: HIT once the array is filled, else return the stored
  //     result (no re-gate).
  auto rit = m_resolved.find(rid);
  if (rit != m_resolved.end()) {
    int way = find_way(sidx, vpn_val);
    if (way >= 0) {
      m_sets[sidx][way].lru_time = cur;
      out_pte = m_sets[sidx][way].pte;
      // Count a hit ONLY for a genuine array-hit classification (HIT).
      // A MISS/MSHR_HIT that has since been filled was already counted in
      // resolve_lookup() (m_misses++ / m_mshr_hits++); counting its completing
      // retry as a hit double-counts every miss and inflates hit_rate.
      if (rit->second == HIT) m_hits++;
      m_resolved.erase(rit);
      // XLATLAT: t9. The PTE is now in the warp's hands, so the request is over.
      auto xr = m_xlat_rec.find(rid);
      if (xr != m_xlat_rec.end()) {
        xr->second.t_done = cur;
        xlat_close(rid, xr->second, cur);  // folds and erases
      }
      return HIT;
    }
    // array miss: if the VPN is still in flight in the MSHR, keep waiting.
    if (m_mshr.probe(vpn_val)) {
      if (rit->second == MISS) rit->second = MSHR_HIT;  // shared VPN entered MSHR
      return rit->second;
    }
    // array miss AND not in flight => the cached classification is STALE:
    // this (warp,vpn) was resolved (HIT/MSHR_HIT) and filled, but the page was
    // later evicted from the small shared L1 array before the warp consumed it.
    // Drop the stale entry and fall through to re-issue a fresh lookup, so the
    // page is re-fetched (L2 hit or re-walk). Otherwise access() returns a stale
    // MSHR_HIT forever and the warp parks on this VPN indefinitely (deadlock).
    m_resolved.erase(rit);
  }
  // (2) Lookup already in the pipe (this request injected): wait, no re-inject.
  //     Idempotent: a re-poll never re-acquires a port or restarts the timer.
  if (m_inflight_lookup.count(rid)) return LOOKUP_PENDING;
  // (2b) Finite lookup ports. Nothing is injected and no resource is held, so the
  //      caller simply retries -- this is a bandwidth limit, not a stall record.
  if (m_n_ports && m_inflight_lookup.size() >= m_n_ports) {
    m_port_busy_stalls++;
    // XLATLAT: the only path where t0 < t1. Poll COUNT, not cycles.
    xlat_req_rec_t &rec = m_xlat_rec[rid];
    if (rec.t_issue == 0) rec.t_issue = cur;
    rec.l1_port_polls++;
    return PORT_BUSY;
  }
  // (3) New: inject at the pipe tail (head reached after lookup_latency cycles).
  m_inflight_lookup.insert(rid);
  if (m_inflight_lookup.size() > m_concurrent_max)
    m_concurrent_max = (unsigned)m_inflight_lookup.size();
  m_lookup_pipe[m_lookup_latency - 1].push_back({true, rid, vpn_val, warp_id});
  // XLATLAT: t1, and t0 as well when the port was free on the first call (the
  // common case -- nothing else can consume a cycle before the injection).
  // Reaching here a second time means block (1) found a stale classification and
  // dropped it: the fill was evicted before the warp could consume it, so the
  // request starts over. Keeping the first injection separate lets S0 absorb the
  // redo without overlapping S1.
  xlat_req_rec_t &rec = m_xlat_rec[rid];
  if (rec.t_issue == 0) rec.t_issue = cur;
  if (rec.t_l1_inject_first == 0) rec.t_l1_inject_first = cur;
  rec.t_l1_inject = cur;
  rec.t_l1_resolve = 0;   // this epoch has not resolved yet
  rec.t_fill = 0;
  return LOOKUP_PENDING;
}

void gpu_l1_tlb::fill(uint64_t vpn, const gpu_pte_t &pte,
                      uint64_t current_cycle, const xlat_down_t *down) {
  m_fill_epoch++;  // 36a: array contents changed; waiters may now hit
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

  // XLATLAT: t8 for every request queued on this VPN. The MSHR already knows the
  // waiter warp ids, so no extra bookkeeping is needed. Stamp before mark_ready:
  // the ldst unit drains the ready list (next_access) in the same cycle and that
  // pops the waiters.
  const std::vector<unsigned> *waiters = m_mshr.waiters_of(vpn);
  if (waiters) {
    for (unsigned w : *waiters) {
      auto r = m_xlat_rec.find(req_key(w, vpn));
      if (r == m_xlat_rec.end()) continue;
      if (r->second.t_fill == 0) r->second.t_fill = current_cycle;
      // Only the request that drove the L1->L2 hand-off owns the downstream
      // path. Attaching `down` to a merged waiter would split its smid with
      // timestamps that predate its own t2 and yield negative segments.
      if (down && r->second.bucket == XLAT_B_PTW && !r->second.has_down) {
        r->second.down = *down;
        r->second.has_down = 1;
      }
    }
  }

  // Mark MSHR entry as ready so tick() can drain pending retries
  m_mshr.mark_ready(vpn);
}

void gpu_l1_tlb::cycle(uint64_t cur) {
  // XLATLAT: sample N once per cycle. size() is O(1), so this costs three adds.
  m_xlat_cycles++;
  const unsigned live = (unsigned)m_xlat_rec.size();
  m_xlat_inflight_sum += live;
  if (live > m_xlat_inflight_max) m_xlat_inflight_max = live;

  std::vector<mshr_wait_ent> this_cycle_blocked;  // 36a: see ORDER note below
  // 36a helper: shared success bookkeeping (identical to the old resolved path).
  auto on_resolved = [&](const lookup_slot &head) {
    m_inflight_lookup.erase(head.rid);
    auto it = m_resv_fail_start.find(head.rid);
    if (it != m_resv_fail_start.end()) {
      m_resv_fail_cycles += (cur - it->second);
      // XLATLAT: sub-attribution inside S2 -- how much of the probe window was
      // spent blocked on an MSHR-full rather than on the array probe itself.
      auto r = m_xlat_rec.find(head.rid);
      if (r != m_xlat_rec.end())
        r->second.l1_resv_fail += (uint32_t)(cur - it->second);
      m_resv_fail_start.erase(it);
    }
  };
  for (lookup_slot &head : m_lookup_pipe[0]) {
    if (!head.valid) continue;
    bool resolved = resolve_lookup(head, cur);
    if (resolved) {
      on_resolved(head);
    } else {
      // 36a: park on the MSHR wait queue instead of re-injecting. Fail type
      // is re-derived cheaply (state is unchanged since the verdict).
      this_cycle_blocked.push_back(
          {head, cur, cur, m_mshr.probe(head.vpn)});
      m_resv_fail_start.emplace(head.rid, cur);
    }
  }
  // 36a ORDER: the old re-inject appended this cycle's failures BEFORE the
  // previously blocked entries in next cycle's head vector (iteration order:
  // arrivals first, then old blocked -- failures collected in that order and
  // appended as a group). Net effect: the most recently blocked GENERATION
  // retries first when a slot frees. Reproduce exactly: prepend this cycle's
  // group, keeping its internal order.
  for (auto it = this_cycle_blocked.rbegin(); it != this_cycle_blocked.rend(); ++it)
    m_mshr_wait_q.push_front(*it);
  for (unsigned s = 0; s + 1 < m_lookup_latency; ++s)
    m_lookup_pipe[s] = std::move(m_lookup_pipe[s + 1]);
  m_lookup_pipe[m_lookup_latency - 1].clear();
  // Q6 fix (kept): an MSHR-full is not a failed probe; the blocked request
  // retries without re-paying m_lookup_latency. 36a changes only WHEN the
  // retry runs: the resolve verdict is a pure function of (array, MSHR)
  // state, so re-attempts happen only on cycles where that state moved
  // (m_fill_epoch / m_mshr.mutations()); on quiet cycles every waiter would
  // fail identically, so skipping them is exact. Waiters run AFTER the
  // arrivals above -- the same priority the old append-behind re-inject gave.
  // entries_full / merge_full are synthesized for the skipped cycles (the
  // fail type cannot change while the state is frozen), keeping the counters
  // bit-identical to the per-cycle retry model.
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
        if (resolve_lookup(w.s, cur)) {
          on_resolved(w.s);
        } else {
          w.last_attempt = cur;
          w.merge_cap = m_mshr.probe(w.s.vpn);
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
  warp_id = m_hit_wakeup.back();
  m_hit_wakeup.pop_back();
  return true;
}

// Reached once per request when its lookup reaches the pipe head.
// Returns false only when the MSHR is full (cycle() re-injects to retry).
bool gpu_l1_tlb::resolve_lookup(const lookup_slot &s, uint64_t cur) {
  unsigned sidx = set_idx(s.vpn);
  // XLATLAT: t2 for every branch below that returns true -- the probe reached the
  // pipe head and the request left the lookup stage. An MSHR-full re-inject
  // (return false) has NOT left it, so it must not be stamped.
  auto xr = m_xlat_rec.find(s.rid);
  int way = find_way(sidx, s.vpn);
  if (way >= 0) {                              // array hit
    m_sets[sidx][way].lru_time = cur;
    m_resolved[s.rid] = HIT;                   // m_hits counted by access() block (1)
    m_resolve_epoch++;                         // 44-5: not covered by fill/MSHR epochs
    m_hit_wakeup.push_back(s.warp_id);      // P5: wake parked warp (no MSHR entry)
    if (xr != m_xlat_rec.end()) {
      xr->second.t_l1_resolve = cur;
      // The page is already in the array, so there is nothing to wait for: t8
      // coincides with t2 and the middle segment is empty by construction.
      xr->second.t_fill  = cur;
      xr->second.bucket = XLAT_B_L1_HIT;
    }
    return true;
  }
  if (m_mshr.probe(s.vpn)) {                    // VPN already in flight
    if (m_mshr.full(s.vpn)) {                   // entry merge cap (waiters >= max_merge)
      m_mshr_merge_full++; return false;        // re-inject; retry until a slot frees / fill lands
    }
    m_mshr.add(s.vpn, s.warp_id);
    m_resolved[s.rid] = MSHR_HIT; m_mshr_hits++;
    if (xr != m_xlat_rec.end()) {
      xr->second.t_l1_resolve = cur;
      xr->second.bucket = XLAT_B_L1_MERGE;
    }
    return true;
  }
  if (m_mshr.full(s.vpn)) {                      // table full (new VPN, no free entry)
    m_mshr_entries_full++; return false;
  }
  m_mshr.add(s.vpn, s.warp_id);                 // add at miss time (capacity-gated)
  if (xr != m_xlat_rec.end()) {
    xr->second.t_l1_resolve = cur;
    xr->second.bucket = XLAT_B_PTW;
  }
  m_l2_req_queue.push_back({s.vpn, s.warp_id, cur, 0, 0, false});  // hand off to L2 fifo (44-3/4: poll now)
  if (m_l2_req_queue.size() > m_l2_req_q_max)
    m_l2_req_q_max = (unsigned)m_l2_req_queue.size();
  m_misses++;
  // Subsequent polls of this same (warp,vpn) request return MSHR_HIT: its
  // L2 request is in flight and the fill will complete it.
  m_resolved[s.rid] = MSHR_HIT;
  return true;
}

// XLATLAT: fold a finished request into its outcome bucket and destroy it.
// A record that never observed every boundary is dropped rather than folded with
// zeroed timestamps, which would silently manufacture a plausible breakdown.
void gpu_l1_tlb::xlat_close(uint64_t rid, xlat_req_rec_t &rec, uint64_t cur) {
  (void)cur;
  if (xlat_rec_complete(rec)) {
    // Refine the provisional L1-level bucket with how the L2 actually answered.
    // Without a `down` record the request never owned the downstream path, so it
    // stays in its L1 bucket and its smid is reported unsplit.
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

// 34: GPU-wide XLATSUM aggregation. print_stats() of every SM folds its
// accumulators in here during a dump; xlatsum_gpu_print() emits the per-outcome
// and ALL rows and resets, so each dump aggregates afresh (raw accumulators
// stay cumulative -- these rows inherit that convention).
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
      all.add(g_xlatsum[k][o]);
      g_xlatsum[k][o] = xlat_sum_t();
    }
    all.print(f, kinds[k], "GPU", "ALL");
  }
}

void gpu_l1_tlb::print_stats(FILE *f) const {
  unsigned long long total = m_hits + m_misses;
  // Tag follows the page size this instance was built for, so 4k and 2m stats
  // stay distinguishable in the log.
  fprintf(f, "[L1_TLB_%s_STAT] SM=%u hits=%llu misses=%llu mshr_hits=%llu "
             "merge_full=%llu entries_full=%llu resv_fail_cycles=%llu "
             "hit_rate=%.2f%%\n",
          page_kind_tag(m_page_shift),
          m_sm_id, m_hits, m_misses, m_mshr_hits,
          m_mshr_merge_full, m_mshr_entries_full, m_resv_fail_cycles,
          total ? 100.0 * m_hits / total : 0.0);
  // Which resource actually limits translation throughput: ports, the L1->L2
  // queue, or something downstream. port_busy==0 means the port cap never bound.
  fprintf(f, "[L1_TLB_%s_PORT] SM=%u ports=%u port_busy=%llu concurrent_max=%u l2_req_q_max=%u\n",
          page_kind_tag(m_page_shift), m_sm_id, m_n_ports,
          m_port_busy_stalls, m_concurrent_max, m_l2_req_q_max);
  // XLATLAT: per-request breakdown. `resid` must be 0; any other value means the
  // segments do not tile [t_issue, t_done] and the whole breakdown is void.
  // `smid` is the not-yet-split middle (L2 + PWQ + PTW + fill delivery).
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
    // Walk-level line. s6/s7/s8 are averaged over n_walk, NOT n: an L2 array hit
    // never reaches the GMMU and is deliberately excluded.
    if (a.n_walk) {
      const double w = (double)a.n_walk;
      fprintf(f,
              "[L1_TLB_%s_XLATWALK] SM=%u out=%s n_walk=%llu s6_pwqwait=%.2f "
              "s7_walk=%.2f s8_deliver=%.2f steps=%.3f remote=%.3f "
              "pwc_hit=%.3f icnt_bp=%.3f\n",
              page_kind_tag(m_page_shift), m_sm_id, xlat_bucket_name(o),
              a.n_walk, a.s6 / w, a.s7 / w, a.s8 / w, a.steps / w,
              a.remote / w, a.pwc_hit / w, a.icnt_bp / w);
      // Shape of S6 and S7. Bin k >= 1 covers [2^(k-1), 2^k) cycles; only
      // non-empty bins are printed.
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
  fprintf(f, "[L1_TLB_%s_XLATLAT] SM=%u dropped=%llu inflight_at_end=%zu\n",
          page_kind_tag(m_page_shift), m_sm_id, m_xlat_dropped,
          m_xlat_rec.size());
  // 34: XLATSUM -- additive summary rows (new tag, outside every digest
  // pattern). Per-SM: one ALL row. Per-outcome rows are emitted GPU-wide by
  // xlatsum_gpu_print() after the SM loop in gpu-sim.cc.
  {
    xlat_sum_t sm_all;
    for (unsigned o = 0; o < XLAT_B_NUM; o++) {
      sm_all.add(m_xlat_acc[o]);
      xlatsum_gpu_fold(page_kind_tag(m_page_shift), o, m_xlat_acc[o]);
    }
    char head[32]; snprintf(head, sizeof(head), "SM=%u", m_sm_id);
    sm_all.print(f, page_kind_tag(m_page_shift), head, "ALL");
  }
  // Little's Law: N (outstanding), X (throughput), W (mean latency). N == X*W is
  // an identity, so a mismatch means a boundary was mismeasured. Reported so the
  // check can be done from the log without re-deriving anything.
  {
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
                                        uint64_t* ready_out) {
  uint64_t vpn_val = vpn(va);
  unsigned sidx    = set_idx(vpn_val);
  uint64_t rid     = req_key(cluster_id, vpn_val);

  // (1) Inside this request's OWN probe-latency window (keyed by rid, not vpn).
  auto git = m_lookup_ready.find(rid);
  if (git != m_lookup_ready.end()) {
    if (cur < git->second) {
      m_gate_pending++;
      if (ready_out) *ready_out = git->second;  // 44-3: let the caller sleep until expiry
      return LOOKUP_PENDING;
    }
    m_gate_expire++;
    m_lookup_ready.erase(git);                        // probe latency elapsed
    result_t r = resolve(vpn_val, sidx, cluster_id, out_pte, cur);
    // Account the time a request stays blocked on an MSHR-full, mirroring
    // gpu_l1_tlb.
    if (r == MSHR_FULL) {
      // Still inside the lookup stage (only the MSHR slot is missing), so the
      // port is NOT released -- mirrors the L1, where an MSHR-full retry keeps
      // its pipe slot.
      m_resv_fail_start.emplace(rid, cur);
      // Q6 fix: re-arm the gate as ALREADY EXPIRED so the next poll retries the
      // MSHR allocation immediately, instead of paying m_lookup_latency (80)
      // again for a probe that already succeeded. Without this the request is
      // re-gated and the stall is amplified 80x per retry.
      m_lookup_ready[rid] = cur;
    } else {
      m_probing.erase(rid);                           // left the lookup stage
      auto fit = m_resv_fail_start.find(rid);
      if (fit != m_resv_fail_start.end()) {
        m_resv_fail_cycles += (cur - fit->second);
        // XLATLAT: sub-attribution inside S4.
        auto xs = m_xlat_seg.find(rid);
        if (xs != m_xlat_seg.end())
          xs->second.l2_resv_fail += (uint32_t)(cur - fit->second);
        m_resv_fail_start.erase(fit);
      }
      // XLATLAT: t4 -- the probe produced a verdict. HIT and MSHR_HIT never reach
      // the PWQ, so t5 collapses onto t4 and S5 is zero for them by construction.
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
    m_xlat_seg[rid].l2_port_polls++;   // XLATLAT: poll COUNT, not cycles
    return PORT_BUSY;
  }
  // (3) Brand-new request: take a port and arm the probe-latency gate.
  m_gate_arm_new++;
  m_probing.insert(rid);
  if (m_probing.size() > m_concurrent_max)
    m_concurrent_max = (unsigned)m_probing.size();
  m_lookup_ready[rid] = cur + m_lookup_latency;
  {
    // XLATLAT: t3. Arming again for a request that already had a verdict means a
    // PWQ-full rollback threw away its gate and port (gpu-module.cc), so the
    // retry pays the full probe latency over again -- count it, but keep t3 at
    // the first arm so S4 covers every re-probe rather than hiding them.
    xlat_down_t& seg = m_xlat_seg[rid];
    if (seg.t_l2_arm == 0) seg.t_l2_arm = cur;
    else                   seg.l2_reprobe++;
  }
  return LOOKUP_PENDING;
}

// XLATLAT: the GMMU accepted this request into the PWQ (t5).
void gpu_l2_tlb::xlat_mark_pwq_enq(unsigned cluster_id, uint64_t vpn_val,
                                   uint64_t cur) {
  auto it = m_xlat_seg.find(req_key(cluster_id, vpn_val));
  if (it != m_xlat_seg.end() && it->second.t_pwq_enq == 0)
    it->second.t_pwq_enq = cur;
}

// XLATLAT: hand the L2 record to the fill path and drop it. Called once per
// (cluster, vpn); a later request for the same page starts a fresh record.
bool gpu_l2_tlb::xlat_take_seg(unsigned cluster_id, uint64_t vpn_val,
                               xlat_down_t& out) {
  auto it = m_xlat_seg.find(req_key(cluster_id, vpn_val));
  if (it == m_xlat_seg.end()) return false;
  out = it->second;
  m_xlat_seg.erase(it);
  return true;
}

// Reached once per gate elapse. Probe the array first, then (on a miss)
// consult the per-VPN MSHR. Merge / entry creation happen HERE -> after latency.
gpu_l2_tlb::result_t
gpu_l2_tlb::resolve(uint64_t vpn_val, unsigned sidx, unsigned cluster_id,
                    gpu_pte_t& out_pte, uint64_t cur) {
  int way = find_way(sidx, vpn_val);
  if (way >= 0) {                                   // array hit
    m_sets[sidx][way].lru_time = cur;
    out_pte = m_sets[sidx][way].pte;
    m_hits++;
    return HIT;
  }
  auto mit = m_mshr.find(vpn_val);
  if (mit != m_mshr.end()) {                        // walk in flight -> merge
    if (mit->second.pending_clusters.size() >= m_mshr_max_merge) {
      m_mshr_merge_full++;                          // entry merge cap
      m_last_full_merge = true;                     // 44-4: fail type for skip synthesis
      return MSHR_FULL;                             // merge cap; re-poll (re-gates)
    }
    mit->second.pending_clusters.push_back(cluster_id);
    m_mshr_epoch++;  // 44-4: merge growth can flip a same-vpn verdict to merge-cap
    m_mshr_hits++;
    return MSHR_HIT;
  }
  if (m_mshr.size() >= m_mshr_entries) {            // MSHR table full
    m_mshr_entries_full++;                          // table full
    m_last_full_merge = false;                      // 44-4: fail type for skip synthesis
    return MSHR_FULL;                               // re-poll (re-gates)
  }
  m_mshr_epoch++;  // 44-4: entry insert changes membership and table occupancy
  l2_mshr_entry_t& e = m_mshr[vpn_val];             // first miss for this vpn
  e.pending_clusters.push_back(cluster_id);
  e.pte_ready = false;
  e.pte.raw   = 0;
  m_misses++;
  return MISS;
}

void gpu_l2_tlb::fill(uint64_t vpn_val, const gpu_pte_t& pte,
                       uint64_t current_cycle, const xlat_walk_t* walk) {
  m_mshr_epoch++;  // 44-4: array contents + pte_ready change verdicts
  // Insert into TLB array with LRU replacement.
  unsigned sidx = set_idx(vpn_val);
  int      way  = find_way(sidx, vpn_val);
  if (way < 0) way = lru_victim(sidx);
  m_sets[sidx][way] = {true, vpn_val, pte, current_cycle};

  // Mark MSHR entry ready if one exists (created on MISS path).
  // HIT path never creates MSHR entries, so this may be a no-op.
  auto it = m_mshr.find(vpn_val);
  if (it != m_mshr.end()) {
    it->second.pte_ready = true;
    it->second.pte       = pte;

    // XLATLAT: the walk's scalars belong to every cluster queued on this VPN --
    // they all waited on the same walk. This is where the requester is known;
    // the GMMU only ever saw the chiplet representative. current_cycle is t7.
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
  if (clusters.empty()) m_mshr.erase(it);  // all SMs consumed; free entry
  m_mshr_epoch++;  // 44-4: consumption shrinks a merge count / frees a slot
  return true;
}

void gpu_l2_tlb::remove_from_mshr(uint64_t vpn_val, unsigned cluster_id) {
  auto it = m_mshr.find(vpn_val);
  if (it == m_mshr.end()) return;
  auto& c = it->second.pending_clusters;
  c.erase(std::remove(c.begin(), c.end(), cluster_id), c.end());
  if (c.empty()) m_mshr.erase(it);
  m_mshr_epoch++;  // 44-4: rollback shrinks a merge count / frees a slot
}

void gpu_l2_tlb::tick(uint64_t /*current_cycle*/) {
  // No-op in Phase 5. Reserved for future fill-latency pipeline drain.
}

void gpu_l2_tlb::print_stats(FILE* f) const {
  unsigned long long total = m_hits + m_misses;
  // Tag follows the page size this instance was built for (see gpu_l1_tlb).
  fprintf(f,
          "[L2_TLB_%s_STAT] chiplet=%u hits=%llu misses=%llu "
          "mshr_hits=%llu merge_full=%llu entries_full=%llu "
          "resv_fail_cycles=%llu hit_rate=%.2f%%\n",
          page_kind_tag(m_page_shift),
          m_chiplet_id, m_hits, m_misses,
          m_mshr_hits, m_mshr_merge_full, m_mshr_entries_full,
          m_resv_fail_cycles,
          total ? 100.0 * m_hits / total : 0.0);
  fprintf(f, "[L2_TLB_%s_PORT] chiplet=%u ports=%u port_busy=%llu concurrent_max=%u\n",
          page_kind_tag(m_page_shift), m_chiplet_id, m_n_ports,
          m_port_busy_stalls, m_concurrent_max);
}

// XLATDBG: the decoupled L1 TLB's in-flight bookkeeping. A request sits in
// m_inflight_lookup from the moment it is injected until its fill arrives, so
// a non-empty set with an empty pipe means the fill never came back.
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
  // The queue is FIFO and only the front is retried, so the front request is the
  // one that must be unblocked -- everything behind it is head-of-line blocked.
  if (!m_l2_req_queue.empty())
    fprintf(fp, " front_vpn=0x%llx front_warp=%u queued_at=%llu",
            (unsigned long long)m_l2_req_queue.front().vpn,
            m_l2_req_queue.front().warp_id,
            (unsigned long long)m_l2_req_queue.front().enqueue_cycle);
  fprintf(fp, "\n");
}

// XLATDBG: why the L2 TLB is turning L1 requests away. Reports occupancy against
// capacity plus the two reject counters, so a merge-cap stall and a table-full
// stall are distinguishable.
void gpu_l2_tlb::xlatdbg_dump(FILE* fp, unsigned chiplet, const char* tag) const {
  fprintf(fp, "XLATDBG_DEADLOCK: l2tlb chiplet=%u %s mshr=%zu/%u lookup_ready=%zu "
              "hits=%llu misses=%llu mshr_hits=%llu merge_full=%llu entries_full=%llu\n",
          chiplet, tag, m_mshr.size(), m_mshr_entries, m_lookup_ready.size(),
          m_hits, m_misses, m_mshr_hits, m_mshr_merge_full, m_mshr_entries_full);
  // arm_new far exceeding the number of distinct requests means gates are being
  // re-armed: each retry pays the full probe latency again.
  fprintf(fp, "XLATDBG_DEADLOCK:   gate: arm_new=%llu pending=%llu expire=%llu\n",
          m_gate_arm_new, m_gate_pending, m_gate_expire);
  // An MSHR entry whose walk finished (pte_ready) but that still has waiters is
  // a fill that never got delivered.
  size_t ready_with_waiters = 0, not_ready = 0;
  for (const auto& kv : m_mshr) {
    if (kv.second.pte_ready && !kv.second.pending_clusters.empty()) ready_with_waiters++;
    if (!kv.second.pte_ready) not_ready++;
  }
  fprintf(fp, "XLATDBG_DEADLOCK:   mshr entries: pte_ready_with_waiters=%zu awaiting_walk=%zu\n",
          ready_with_waiters, not_ready);
}

// XLATDBG: what access() would answer for `vpn` right now, without mutating.
void gpu_l2_tlb::xlatdbg_probe(FILE* fp, uint64_t vpn, unsigned cluster_id,
                               uint64_t cur) const {
  const uint64_t rid = req_key(cluster_id, vpn);
  auto rit = m_lookup_ready.find(rid);
  const char* gate = "no_gate";
  unsigned long long ready_at = 0;
  if (rit != m_lookup_ready.end()) {
    ready_at = (unsigned long long)rit->second;
    gate = (cur < rit->second) ? "LOOKUP_PENDING" : "gate_expired";
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
