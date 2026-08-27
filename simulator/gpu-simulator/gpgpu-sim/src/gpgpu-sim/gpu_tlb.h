#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "gpu-module.h"  // gpu_pte_t, pt_vpn(), GPU_PAGE_SHIFT
#include "xlat_stat.h"   // XLATLAT: per-request latency breakdown records
#include <unordered_set>
#include <deque>

//  L1 TLB MSHR
// Tracks in-flight GMMU translation requests, keyed by VPN.
// Does NOT store mem_fetch* -- the park/replay mechanism handles wakeup.
// Mirrors mshr_table (gpu-cache.h) API: probe/full/add/mark_ready.
class l1_tlb_mshr {
 public:
  l1_tlb_mshr(unsigned num_entries, unsigned max_merge);

  // Returns true if there is already an outstanding request for vpn.
  bool probe(uint64_t vpn) const;

  // Returns true if MSHR is full for vpn (either table full or max_merge hit).
  bool full(uint64_t vpn) const;

  // Add one pending request for vpn. Caller must check !full() first.
  struct l1_fill_token { uint64_t vpn; std::vector<unsigned> waiters; };
  void add(uint64_t vpn, unsigned warp_id);

  // 36a: bumped by every mutation that can change a resolve_lookup verdict
  // (add / next_access / evict). The MSHR-full wait queue re-attempts only
  // when this moves, instead of every cycle.
  uint64_t mutations() const { return m_mutations; }
  uint64_t m_mutations = 0;  // 36a


  // Mark vpn as ready (GMMU done). Does not remove entry ? caller removes via evict().
  void mark_ready(uint64_t vpn);

  // Returns true if there is at least one ready (GMMU done) VPN.
  bool access_ready() const;

  // Remove and return one ready VPN (for TLB fill).
  l1_fill_token next_access();   // pop one ready vpn + its waiters, erase entry

  // Remove entry for vpn entirely (after TLB fill completes).
  void evict(uint64_t vpn);

  // XLATLAT: the warps queued on this VPN, so a fill can stamp t8 on each of
  // their records. Returns NULL when no entry exists (an L1 array hit never
  // allocates one). Exposes existing state; nothing extra is tracked.
  const std::vector<unsigned> *waiters_of(uint64_t vpn) const {
    auto it = m_entries.find(vpn);
    return (it == m_entries.end()) ? NULL : &it->second.waiters;
  }

  unsigned size() const { return (unsigned)m_entries.size(); }

  std::vector<uint64_t> get_pending_vpns() const;

 private:
  unsigned m_num_entries;
  unsigned m_max_merge;

  struct mshr_entry_t {
    std::vector<unsigned> waiters;   // merged warp_ids
    bool ready;
  };

  std::unordered_map<uint64_t, mshr_entry_t> m_entries;
  std::vector<uint64_t>                      m_ready_list;
};

//  L1 TLB set-associative entry 
struct l1_tlb_entry_t {
  bool      valid;
  uint64_t  vpn;
  gpu_pte_t pte;
  uint64_t  lru_time;
};

//  L1 TLB 
// Per-SM, set-associative, LRU replacement.
// Capacity = num_sets * assoc entries.
// 34: XLATSUM GPU-wide aggregation (definitions in gpu_tlb.cc).
struct xlat_accum_t;
void xlatsum_gpu_fold(const char* kind, unsigned outcome, const struct xlat_accum_t &a);
void xlatsum_gpu_print(FILE *f);

class gpu_l1_tlb {
 public:
  // num_sets: number of sets (config: -gpgpu_l1_tlb_sets)
  // assoc   : ways per set   (config: -gpgpu_l1_tlb_assoc)
  // mshr_entries: MSHR table capacity  (config: -gpgpu_l1_tlb_mshr_entries)
  // mshr_max_merge: max pending per VPN (config: -gpgpu_l1_tlb_mshr_max_merge)
  // sm_id: owner SM index (used for logging)
  gpu_l1_tlb(unsigned num_sets, unsigned assoc,
             unsigned mshr_entries, unsigned mshr_max_merge,
             unsigned lookup_latency,
             unsigned sm_id,
             unsigned page_shift = GPU_PAGE_SHIFT);

  enum result_t { HIT, MISS, MSHR_HIT, LOOKUP_PENDING, PORT_BUSY };

  // Lookup va in the TLB.
  // HIT   : out_pte filled, return HIT.
  // MISS  : vpn not in TLB and not in MSHR -> caller must enqueue GMMU.
  // MSHR_HIT: vpn already in-flight -> stall (no new GMMU request).
  // LOOKUP_PENDING: vpn is in the lookup queue but not yet ready.
  result_t access(uint64_t va, unsigned warp_id,
                  gpu_pte_t &out_pte, uint64_t current_cycle);

  // Insert (or update) a TLB entry from GMMU fill.
  // XLATLAT: `down` carries the L2/GMMU boundaries measured for the (cluster,vpn)
  // request that drove this fill. It is attached only to the L1-miss request that
  // owns that path -- merged waiters would otherwise be split with someone else's
  // timestamps and produce negative segments.
  void fill(uint64_t vpn, const gpu_pte_t &pte, uint64_t current_cycle,
            const xlat_down_t *down = NULL);

  // Called each cycle: advance lookup pipe, resolve head, drain MSHR-ready.
  void cycle(uint64_t current_cycle);
  // L1->L2 request queue accessors (drained by ldst_unit::cycle).
  bool l2_req_front(uint64_t &vpn) const;
  void l2_req_pop();
  // P5: warps whose lookup resolved to a cached HIT (no MSHR entry) -> wake/replay.
  bool pop_hit_wakeup(unsigned &warp_id);


  l1_tlb_mshr &mshr() { return m_mshr; }

  // verification accessor (unguarded on purpose: gpu_tlb.h is pulled in via
  // shader.h BEFORE gmmu.h defines ADDR_XLAT_VERIFY, so an #ifdef here would
  // be invisible; this tiny inline getter is zero-cost when verify is off)
  void set_ports(unsigned p) { m_n_ports = p; }
  unsigned long long port_busy_stalls() const { return m_port_busy_stalls; }
  unsigned concurrent_max() const { return m_concurrent_max; }
  unsigned long long xlat_misses() const { return m_misses; } // ADDR_XLAT_VERIFY

  // STATEHASH: order-independent fingerprint of this TLB's mutable state.
  // Compared cycle by cycle between a 1-thread and an N-thread run to find the
  // FIRST cycle at which the two trajectories differ -- before the divergence
  // spreads and every statistic looks wrong. Sums rather than sequences, so the
  // value never depends on container iteration order (itself an artefact).
  unsigned long long state_fingerprint() const {
    unsigned long long h = 1469598103934665603ull;
    auto mix = [&h](unsigned long long v) { h ^= v; h *= 1099511628211ull; };
    mix(m_hits); mix(m_misses);
    mix(m_inflight_lookup_4k.size()); mix(m_resolved.size());
    mix(m_l2_req_queue_4k.size()); mix(m_mshr.size());
    unsigned long long tag_sum = 0, valid = 0;
    for (const auto &set : m_sets)
      for (const auto &e : set) if (e.valid) { valid++; tag_sum += e.vpn; }
    mix(valid); mix(tag_sum);
    unsigned long long pipe = 0;
    for (const auto &stage : m_lookup_pipe_4k) pipe += stage.size();
    mix(pipe);
    return h;
  }

  // XLATDBG: how many lookups/fills are outstanding at deadlock time.
  void xlatdbg_dump(FILE* fp, unsigned sid, const char* tag) const;

  void print_stats(FILE *f = stdout) const;

 private:
  unsigned m_num_sets;
  unsigned m_assoc;
  unsigned m_sm_id;
  unsigned m_page_shift;   // added: used by vpn() and set_idx()
  unsigned m_lookup_latency;

  // [set_idx][way]
  std::vector<std::vector<l1_tlb_entry_t>> m_sets;

  l1_tlb_mshr m_mshr;
  struct lookup_slot { bool valid; uint64_t rid; uint64_t vpn; unsigned warp_id; };
  // 36a: entries whose resolve failed on an MSHR condition (table full or
  // merge cap). They leave the pipe and wait here; re-attempted ONLY on cycles
  // where the TLB state moved (fill or MSHR mutation) -- the verdict is a pure
  // function of that state, so skipping unchanged cycles is exact. Stat
  // synthesis keeps entries_full / merge_full identical to the per-cycle
  // retry model (fail type is constant between state changes by the same
  // purity argument; dumps happen at kernel ends where this queue is empty).
  struct mshr_wait_ent {
    lookup_slot s;
    uint64_t enq;           // first-fail cycle (t of the arrival attempt)
    uint64_t last_attempt;  // last real resolve attempt
    bool merge_cap;         // fail type at last attempt (false = table full)
  };
  std::deque<mshr_wait_ent> m_mshr_wait_q;
  uint64_t m_fill_epoch = 0;       // bumped by fill()
  uint64_t m_last_wake_epoch = ~0ull;
  // 44-5: bumped when resolve_lookup records a verdict that neither fill()
  // nor an MSHR mutation covers (the array-hit resolve writes m_resolved and
  // wakes the warp without touching either).
  uint64_t m_resolve_epoch = 0;
 public:
  // 44-5: moves whenever anything that can change an in-flight request's
  // access() answer moves. The replay re-probe scan is skipped while this
  // (summed over the SM's TLBs) stands still.
  uint64_t change_epoch() const {
    return m_fill_epoch + m_mshr.m_mutations + m_resolve_epoch;
  }
  unsigned ports() const { return m_n_ports; }
 private:
  // NOTE on the _4k suffixes below (findings.ko.md SMELL-10): historical names.
  // Every gpu_l1_tlb INSTANCE (4k, 64k and 2m alike) uses these members for its
  // own page size; nothing here is 4KB-specific. Index = remaining latency stage.
  std::vector<std::vector<lookup_slot>> m_lookup_pipe_4k;
  // Lookup ports: cap on how many probes may sit in the pipe at once.
  // 0 = unlimited (pre-existing behaviour). m_inflight_lookup_4k IS the
  // occupancy set, so there is no second counter to drift out of sync.
  unsigned m_n_ports = 0;
  unsigned long long m_port_busy_stalls = 0;
  unsigned m_concurrent_max = 0;
  unsigned m_l2_req_q_max = 0;
  std::unordered_set<uint64_t> m_inflight_lookup_4k;  // rid: prevent re-injection (re-poll idempotent)
  std::unordered_map<uint64_t, result_t> m_resolved;  // rid -> MISS / MSHR_HIT (awaiting fill)

  unsigned long long m_hits;
  unsigned long long m_misses;
  unsigned long long m_mshr_hits;
  unsigned long long m_mshr_merge_full   = 0;  // entry merge cap hit (waiters >= max_merge)
  unsigned long long m_mshr_entries_full = 0;  // MSHR table full (no free entry for a new VPN)
  unsigned long long m_resv_fail_cycles  = 0;  // total cycles requests stayed blocked (both kinds)
  std::unordered_map<uint64_t, uint64_t> m_resv_fail_start;  // rid -> first cycle it hit MSHR-full
  // m_resv_fail_events removed -- it equals merge_full + entries_full now

  // XLATLAT: in-flight per-request records, keyed by the same rid as the lookup
  // pipe. A record is created on the first access() for a (warp, vpn) and erased
  // the moment that request hands its PTE to the warp.
  std::unordered_map<uint64_t, xlat_req_rec_t> m_xlat_rec;
  xlat_accum_t       m_xlat_acc[XLAT_B_NUM];
  unsigned long long m_xlat_dropped = 0;  // records abandoned without completing

  // XLATLAT: Little's Law inputs, sampled once per cycle. m_xlat_rec holds
  // exactly the requests between t0 and t9, so its size IS N (outstanding
  // translations). With X = completions/cycle and W = mean total latency, the
  // identity N = X*W must hold -- an independent check on the whole breakdown.
  unsigned long long m_xlat_cycles = 0;
  unsigned long long m_xlat_inflight_sum = 0;
  unsigned           m_xlat_inflight_max = 0;

  void xlat_close(uint64_t rid, xlat_req_rec_t &rec, uint64_t cur);

  
  uint64_t vpn(uint64_t va)      const { return va >> m_page_shift; }
  unsigned set_idx(uint64_t vpn) const { return vpn % m_num_sets; }
  static uint64_t req_key(unsigned warp_id, uint64_t vpn) {  // add
    return (vpn << 8) | (warp_id & 0xFF);                    // warp_id < 256
  }
  int find_way(unsigned sidx, uint64_t vpn) const;
  int lru_victim(unsigned sidx) const;
  bool resolve_lookup(const lookup_slot &s, uint64_t cur);

  // 44-3: next_poll = the L2 probe-gate expiry for this entry (0 = poll now).
  // While the gate is open the L2 answer is LOOKUP_PENDING with no state
  // change, so the drain loop skips the call entirely until that cycle.
  // 44-4: wait_epoch != 0 means the last poll answered MSHR_FULL; the verdict
  // is a pure function of the L2 (array, MSHR) state, so re-poll only when the
  // L2 state epoch moves. wait_merge remembers which counter the frozen
  // verdict was charging (merge cap vs table full) for exact synthesis.
  struct l1_tlb_req { uint64_t vpn; unsigned warp_id; uint64_t enqueue_cycle;
                      uint64_t next_poll; uint64_t wait_epoch; bool wait_merge; };
  std::deque<l1_tlb_req> m_l2_req_queue_4k;   // cache: m_miss_queue
  std::vector<unsigned>  m_hit_wakeup_4k;     // P5: parked warps resolved to HIT

 public:
  // Iterate the pending-to-L2 queue. The drain must be able to issue past a
  // request the L2 is still probing, so head-only access is not enough.
  // (Declared here because it names l1_tlb_req, defined just above.)
  typedef std::deque<l1_tlb_req>::iterator l2_req_iter;
  l2_req_iter l2_req_begin() { return m_l2_req_queue_4k.begin(); }
  l2_req_iter l2_req_end()   { return m_l2_req_queue_4k.end(); }
  l2_req_iter l2_req_erase(l2_req_iter it) { return m_l2_req_queue_4k.erase(it); }
};

//  L2 TLB MSHR entry
// Unlike L1 MSHR (which tracks warp count per VPN within one SM), L2 MSHR
// tracks which cluster_ids across multiple SMs are waiting for a VPN.
// The completed pte is stored here so each SM can independently poll
// and retrieve the result via query_completed() at different cycles.
struct l2_mshr_entry_t {
  std::vector<unsigned> pending_clusters;  // cluster_ids (tpc) awaiting this VPN
  bool      pte_ready;  // set true by fill() when GMMU walk completes
  gpu_pte_t pte;        // result pte, valid when pte_ready == true
};

//  L2 TLB
// Per-chiplet, set-associative, LRU replacement.
// Shared by all SMs on the chiplet; deduplicates PTW requests via L2 MSHR.
// Interface mirrors gpu_l1_tlb: access / fill / tick / print_stats.
class gpu_l2_tlb {
 public:
  // num_sets    : number of sets  (config: -gpgpu_l2_tlb_sets)
  // assoc       : ways per set    (config: -gpgpu_l2_tlb_assoc)
  // mshr_entries: max unique VPNs in-flight simultaneously
  //               (config: -gpgpu_l2_tlb_mshr_entries)
  // chiplet_id  : owner chiplet (for logging)
  gpu_l2_tlb(unsigned num_sets, unsigned assoc,
             unsigned mshr_entries, unsigned mshr_max_merge,
             unsigned lookup_latency, unsigned chiplet_id,
             unsigned page_shift = GPU_PAGE_SHIFT);

  // HIT:       VPN found in TLB array. out_pte filled. No MSHR involvement.
  //            Mirrors gpu_l1_tlb::access() HIT behavior.
  // MISS:      VPN absent. New L2 MSHR entry created (pte_ready=false).
  //            Caller (gpu_module::l2_tlb_request) must call gmmu->translate().
  // MSHR_HIT:  VPN already in-flight. cluster_id appended to pending_clusters.
  //            No new gmmu->translate() call needed.
  // MSHR_FULL: New VPN but L2 MSHR at capacity. No entry created.
  //            Caller returns false so SM retries next cycle.
  enum result_t { HIT, MISS, MSHR_HIT, MSHR_FULL, LOOKUP_PENDING, PORT_BUSY };

  // 44-3: ready_out (optional) reports the probe-gate expiry cycle when the
  // answer is LOOKUP_PENDING, so the drain loop can skip re-polls until then.
  result_t access(uint64_t va, unsigned cluster_id,
                  gpu_pte_t& out_pte, uint64_t current_cycle,
                  uint64_t* ready_out = NULL);

  // XLATLAT: L2-side boundaries for one (cluster, vpn) request.
  //   mark_pwq_enq : the GMMU accepted it into the PWQ (t5)
  //   take_seg     : hand the record up and drop it (called once, at fill time)
  // A PWQ-full rollback releases both the gate and the port, so the retry pays a
  // fresh full-latency probe; mark_reprobe() counts those.
  void xlat_mark_pwq_enq(unsigned cluster_id, uint64_t vpn_val, uint64_t cur);
  bool xlat_take_seg(unsigned cluster_id, uint64_t vpn_val, xlat_down_t& out);

  // Called by gmmu::complete_walk() when PTW finishes.
  // Inserts pte into TLB array (LRU eviction) and marks L2 MSHR pte_ready.
  // XLATLAT: `walk` holds the walk's scalars and current_cycle is t7. The GMMU
  // cannot name the requesting clusters, but the MSHR entry can, so the stamping
  // happens here.
  void fill(uint64_t vpn, const gpu_pte_t& pte, uint64_t current_cycle,
            const struct xlat_walk_t* walk = NULL);

  // Called per SM in ldst_unit::cycle() polling loop.
  // If pte_ready and cluster_id is in pending_clusters: fills out_pte, removes
  // cluster_id (frees entry when list empties), returns true.
  bool query_completed(uint64_t vpn, unsigned cluster_id, gpu_pte_t& out_pte);

  // Removes cluster_id from MSHR entry for vpn.
  // Used by l2_tlb_request() to roll back when GMMU PWQ is full.
  void set_ports(unsigned p) { m_n_ports = p; }
  unsigned long long port_busy_stalls() const { return m_port_busy_stalls; }
  unsigned concurrent_max() const { return m_concurrent_max; }

  // STATEHASH: see gpu_l1_tlb::state_fingerprint().
  unsigned long long state_fingerprint() const {
    unsigned long long h = 1469598103934665603ull;
    auto mix = [&h](unsigned long long v) { h ^= v; h *= 1099511628211ull; };
    mix(m_hits); mix(m_misses); mix(m_mshr_hits);
    mix(m_probing.size()); mix(m_mshr.size());
    unsigned long long tag_sum = 0, valid = 0;
    for (const auto &set : m_sets)
      for (const auto &e : set) if (e.valid) { valid++; tag_sum += e.vpn; }
    mix(valid); mix(tag_sum);
    unsigned long long mshr_key_sum = 0;
    for (const auto &kv : m_mshr) mshr_key_sum += kv.first;
    mix(mshr_key_sum);
    return h;
  }

  // Is this request already occupying a lookup port (being probed)?
  bool is_probing(unsigned cluster_id, uint64_t vpn_val) const {
    return m_probing.count(req_key(cluster_id, vpn_val)) != 0;
  }
  void remove_from_mshr(uint64_t vpn, unsigned cluster_id);
  // XLATDBG: L2 MSHR occupancy and the reason it rejects, at deadlock time.
  void xlatdbg_dump(FILE* fp, unsigned chiplet, const char* tag) const;
  // XLATDBG: non-mutating probe -- what would access() answer for this vpn?
  void xlatdbg_probe(FILE* fp, uint64_t vpn, unsigned cluster_id, uint64_t cur) const;

  // No-op in Phase 5; reserved for future fill-latency pipeline modeling.
  void tick(uint64_t current_cycle);

  void print_stats(FILE* f = stdout) const;

 private:
  unsigned m_num_sets;
  unsigned m_assoc;
  unsigned m_chiplet_id;
  unsigned m_page_shift;
  // Lookup ports. A port is held from injection until the request leaves the
  // lookup stage; an MSHR-full retry has not left it yet, so it keeps its slot.
  // 0 = unlimited, which is the behaviour that existed before this option.
  unsigned m_n_ports = 0;
  std::unordered_set<uint64_t> m_probing;      // rids holding a port
  unsigned long long m_port_busy_stalls = 0;
  unsigned m_concurrent_max = 0;
  unsigned m_mshr_entries;
  unsigned m_mshr_max_merge;
  unsigned m_lookup_latency;

  std::vector<std::vector<l1_tlb_entry_t>> m_sets;  // reuse L1 entry type
  std::unordered_map<uint64_t, l2_mshr_entry_t> m_mshr;
  std::unordered_map<uint64_t, uint64_t> m_lookup_ready;  // keyed by req_key(cluster,vpn)

  unsigned long long m_hits;
  unsigned long long m_misses;
  unsigned long long m_mshr_hits;
  unsigned long long m_mshr_merge_full   = 0;  // pending_clusters >= max_merge (entry merge cap)
  unsigned long long m_mshr_entries_full = 0;  // MSHR table full (new VPN, no free entry)
  // Reservation-fail accounting, mirroring gpu_l1_tlb. The L2 had none even
  // though its probe latency (80) makes an MSHR-full stall far costlier.
  // XLATDBG: which gate branch each access() took.
  unsigned long long m_gate_arm_new = 0;   // branch (2): gate absent -> arm cur+latency
  unsigned long long m_gate_pending = 0;   // branch (1): still inside the window
 public:
  // 44-3: the drain loop skips the whole l2_tlb_request() call while the gate
  // is open. The original counted one LOOKUP_PENDING per reached entry per
  // cycle; the skip path reproduces exactly that count here.
  void note_gate_pending() { m_gate_pending++; }
  // 44-4: state epoch for the MSHR_FULL retry gate. Bumped by every mutation
  // that can change a resolve() verdict or its fail type: fill (array + MSHR),
  // MSHR entry insert/erase, and pending_clusters growth/consumption.
  uint64_t mshr_epoch() const { return m_mshr_epoch; }
  bool last_full_merge() const { return m_last_full_merge; }
  // One skipped reached-cycle == one original retry: it charged gate_expire
  // plus the (frozen) fail-type counter inside resolve().
  void note_mshr_full_skip(bool merge) {
    m_gate_expire++;
    if (merge) m_mshr_merge_full++; else m_mshr_entries_full++;
  }
 private:
  uint64_t m_mshr_epoch = 1;
  bool m_last_full_merge = false;

  unsigned long long m_gate_expire  = 0;   // branch (1): window elapsed -> resolve()
  unsigned long long m_resv_fail_cycles = 0;
  std::unordered_map<uint64_t, uint64_t> m_resv_fail_start;   // rid -> first fail cycle

  // XLATLAT: L2 boundaries per in-flight (cluster, vpn). Erased when the fill
  // path takes it, so it never outlives the request it describes.
  std::unordered_map<uint64_t, xlat_down_t> m_xlat_seg;


  uint64_t vpn(uint64_t va)      const { return va >> m_page_shift; }
  unsigned set_idx(uint64_t vpn) const { return vpn % m_num_sets; }
  static uint64_t req_key(unsigned cluster_id, uint64_t vpn) {  // add
    return (vpn << 12) | (cluster_id & 0xFFF);                  // cluster_id < 4096
  }
  result_t resolve(uint64_t vpn_val, unsigned sidx, unsigned cluster_id,  // add
                   gpu_pte_t& out_pte, uint64_t cur);
  int find_way(unsigned sidx, uint64_t vpn) const;
  int lru_victim(unsigned sidx) const;
};
