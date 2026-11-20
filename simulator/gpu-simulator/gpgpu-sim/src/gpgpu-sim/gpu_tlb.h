#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "gpu-module.h"  // gpu_pte_t, pt_vpn(), GPU_PAGE_SHIFT
#include "xlat_stat.h"   // per-request translation latency breakdown records
#include <unordered_set>
#include <deque>

// L1 TLB MSHR: in-flight translations keyed by VPN; holds warp ids, not mem_fetch*.
class l1_tlb_mshr {
 public:
  l1_tlb_mshr(unsigned num_entries, unsigned max_merge);

  // Returns true if there is already an outstanding request for vpn.
  bool probe(uint64_t vpn) const;

  // Returns true if MSHR is full for vpn (either table full or max_merge hit).
  bool full(uint64_t vpn) const;

  // probe(vpn) and full(vpn) in one hash lookup. Read-only.
  void peek(uint64_t vpn, bool &in_flight, bool &is_full) const {
    auto it = m_entries.find(vpn);
    in_flight = (it != m_entries.end());
    is_full = in_flight ? (it->second.waiters.size() >= m_max_merge)
                        : (m_entries.size() >= m_num_entries);
  }

  // Add one pending request for vpn. Caller must check !full() first.
  struct l1_fill_token { uint64_t vpn; std::vector<unsigned> waiters; };
  void add(uint64_t vpn, unsigned warp_id);

  // Bumped by each change that can flip a resolve verdict (add, next_access, evict).
  uint64_t mutations() const { return m_mutations; }
  uint64_t m_mutations = 0;


  // Mark vpn as ready (GMMU done). Does not remove the entry; next_access() does.
  void mark_ready(uint64_t vpn);

  // Returns true if there is at least one ready (GMMU done) VPN.
  bool access_ready() const;

  // Remove and return one ready VPN (for TLB fill).
  l1_fill_token next_access();   // pop one ready vpn + its waiters, erase entry

  // Remove the entry for vpn without a fill (verification harness only).
  void evict(uint64_t vpn);

  // Warps queued on vpn (a fill hands each the PTE); NULL if there is no entry.
  const std::vector<unsigned> *waiters_of(uint64_t vpn) const {
    auto it = m_entries.find(vpn);
    return (it == m_entries.end()) ? NULL : &it->second.waiters;
  }

  unsigned size() const { return (unsigned)m_entries.size(); }

  // True if vpn has an entry whose walk has not completed.
  bool is_pending(uint64_t vpn) const {
    auto it = m_entries.find(vpn);
    return it != m_entries.end() && !it->second.ready;
  }

  // Visit each VPN whose walk is not done; f may call mark_ready() but not add/erase entries.
  template <class F> void for_each_pending(F f) {
    for (auto &kv : m_entries)
      if (!kv.second.ready) f(kv.first);
  }

 private:
  unsigned m_num_entries;
  unsigned m_max_merge;

  struct mshr_entry_t {
    std::vector<unsigned> waiters;   // merged warp_ids
    bool ready = false;
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

// GPU-wide [XLATSUM] aggregation (definitions in gpu_tlb.cc).
struct xlat_accum_t;
void xlatsum_gpu_fold(const char* kind, unsigned outcome, const struct xlat_accum_t &a);
void xlatsum_gpu_print(FILE *f);

// L1 TLB: per SM, set-associative, LRU.
class gpu_l1_tlb {
 public:
  // Sizes are the -gpgpu_l1_tlb_<4k|64k|2m>_* options; sm_id is used only in logs.
  gpu_l1_tlb(unsigned num_sets, unsigned assoc,
             unsigned mshr_entries, unsigned mshr_max_merge,
             unsigned lookup_latency,
             unsigned sm_id,
             unsigned page_shift = GPU_PAGE_SHIFT);

  enum result_t { HIT, MISS, MSHR_HIT, LOOKUP_PENDING, PORT_BUSY };

  // Look up va. HIT: out_pte filled. MSHR_HIT: walk in flight, wait.
  // LOOKUP_PENDING: lookup not done yet. MISS is never returned (a miss reads MSHR_HIT).
  result_t access(uint64_t va, unsigned warp_id,
                  gpu_pte_t &out_pte, uint64_t current_cycle);

  // Insert or update an entry from a GMMU fill; `down` goes only to the request that missed in L2.
  void fill(uint64_t vpn, const gpu_pte_t &pte, uint64_t current_cycle,
            const xlat_down_t *down = NULL);

  // Per cycle: resolve the pipe head, advance it, start queued lookups, retry MSHR-full waiters.
  void cycle(uint64_t current_cycle);
  // L1->L2 request queue accessors (drained by ldst_unit::cycle).
  bool l2_req_front(uint64_t &vpn) const;
  void l2_req_pop();
  // Warps whose lookup resolved to a cached HIT (no MSHR entry) -> wake/replay.
  bool pop_hit_wakeup(unsigned &warp_id);


  l1_tlb_mshr &mshr() { return m_mshr; }

  // Request keys hold this many warp-id bits (checked against the warp count at startup).
  static const unsigned kReqKeyWarpBits = 8;
  void set_ports(unsigned p) { m_n_ports = p; }
  void set_lookups_per_cycle(unsigned n) { m_lookups_per_cycle = n; }
  void set_mshr_wait_fifo(bool on) { m_mshr_wait_fifo = on; }
  // -gpgpu_l1d_vipt: a hit overlaps the L1D access (see access()).
  void set_hit_overlap(bool on) { m_hit_overlap = on; }
  unsigned long long port_busy_stalls() const { return m_port_busy_stalls; }
  unsigned concurrent_max() const { return m_concurrent_max; }
  // Not under #ifdef: ADDR_XLAT_VERIFY is defined only after this header.
  unsigned long long xlat_misses() const { return m_misses; } // ADDR_XLAT_VERIFY

  // STATEHASH: order-independent state fingerprint, compared per cycle across thread counts.
  unsigned long long state_fingerprint() const {
    unsigned long long h = 1469598103934665603ull;
    auto mix = [&h](unsigned long long v) { h ^= v; h *= 1099511628211ull; };
    mix(m_hits); mix(m_misses);
    mix(m_inflight_lookup.size()); mix(m_resolved.size());
    mix(m_l2_req_queue.size()); mix(m_mshr.size());
    unsigned long long tag_sum = 0, valid = 0;
    for (const auto &set : m_sets)
      for (const auto &e : set) if (e.valid) { valid++; tag_sum += e.vpn; }
    mix(valid); mix(tag_sum);
    unsigned long long pipe = 0;
    for (const auto &stage : m_lookup_pipe) pipe += stage.size();
    mix(pipe);
    mix(m_inject_q.size());
    mix(m_hit_wakeup.size());
    return h;
  }

  // Deadlock dump (XLATDBG_DEADLOCK rows): how many lookups/fills are outstanding.
  void xlatdbg_dump(FILE* fp, unsigned sid, const char* tag) const;

  void print_stats(FILE *f = stdout) const;

 private:
  unsigned m_num_sets;
  unsigned m_assoc;
  unsigned m_sm_id;
  unsigned m_page_shift;   // used by vpn() and set_idx()
  unsigned m_lookup_latency;

  // [set_idx][way]
  std::vector<std::vector<l1_tlb_entry_t>> m_sets;

  l1_tlb_mshr m_mshr;
  struct lookup_slot { bool valid; uint64_t rid; uint64_t vpn; unsigned warp_id; };
  // Lookups blocked on a full MSHR; retried only when a fill or MSHR change could alter the verdict.
  struct mshr_wait_ent {
    lookup_slot s;
    uint64_t enq;           // first-fail cycle (t of the arrival attempt)
    uint64_t last_attempt;  // last real resolve attempt
    bool merge_cap;         // fail type at last attempt (false = table full)
  };
  std::deque<mshr_wait_ent> m_mshr_wait_q;
  uint64_t m_fill_epoch = 0;       // bumped by fill()
  uint64_t m_last_wake_epoch = ~0ull;
  // Translation-scan memo: bumped by an array-hit resolve (it moves no fill/MSHR epoch).
  uint64_t m_resolve_epoch = 0;
 public:
  // Translation-scan memo: moves when any in-flight access() answer may change.
  uint64_t change_epoch() const {
    return m_fill_epoch + m_mshr.m_mutations + m_resolve_epoch;
  }
  unsigned ports() const { return m_n_ports; }
 private:
  // Lookup pipe: one vector per latency stage, used as a ring.
  std::vector<std::vector<lookup_slot>> m_lookup_pipe;
  // Ring head: stage k (k cycles left) is pipe_stage(k); cycle() drains stage 0, then advances.
  unsigned m_pipe_head = 0;
  // Slots in the whole ring; when 0 and nothing waits, cycle() returns early.
  unsigned m_pipe_live = 0;
  std::vector<lookup_slot> &pipe_stage(unsigned k) {
    return m_lookup_pipe[(m_pipe_head + k) % m_lookup_latency];
  }
  // Lookup ports: max lookups in flight (m_inflight_lookup.size()); 0 = unlimited.
  unsigned m_n_ports = 0;
  unsigned long long m_port_busy_stalls = 0;
  // Lookups started per cycle; 0 = unlimited. Extra requests wait in m_inject_q (FIFO).
  unsigned m_lookups_per_cycle = 0;
  bool m_hit_overlap = false;  // -gpgpu_l1d_vipt
  bool m_mshr_wait_fifo = false;  // -gpgpu_l1_tlb_mshr_wait_fifo
  uint64_t m_inject_cycle = ~0ull;   // cycle that m_inject_count refers to
  unsigned m_inject_count = 0;       // lookups started in m_inject_cycle
  std::deque<lookup_slot> m_inject_q;
  unsigned long long m_inject_queued = 0;  // requests that had to wait for a slot
  unsigned m_inject_q_max = 0;
  bool take_lookup_slot(uint64_t cur);
  void start_lookup(const lookup_slot &s, uint64_t cur);
  unsigned m_concurrent_max = 0;
  unsigned m_l2_req_q_max = 0;
  std::unordered_set<uint64_t> m_inflight_lookup;  // rid: prevent re-injection (re-poll idempotent)
  // A resolved request keeps its delivered PTE, so an eviction before replay does not lose it.
  struct resolved_t { result_t r; bool has_pte; gpu_pte_t pte; };
  std::unordered_map<uint64_t, resolved_t> m_resolved;  // rid -> verdict (+ PTE once delivered)
  unsigned long long m_fwd_delivered = 0;  // consumptions served from the delivered PTE

  unsigned long long m_hits;
  unsigned long long m_misses;
  unsigned long long m_mshr_hits;
  // MSHR-full counts are blocked cycles by cause: merge_full + entries_full == resv_fail_cycles.
  unsigned long long m_mshr_merge_full   = 0;  // cycles blocked by the per-entry merge cap
  unsigned long long m_mshr_entries_full = 0;  // cycles blocked by a full MSHR table (new VPN)
  unsigned long long m_resv_fail_cycles  = 0;  // sum of the two (kept for log compatibility)
  std::unordered_map<uint64_t, uint64_t> m_resv_fail_start;  // rid -> first cycle it hit MSHR-full

  // Latency breakdown: one record per in-flight request (rid), from first access() to PTE handoff.
  std::unordered_map<uint64_t, xlat_req_rec_t> m_xlat_rec;
  xlat_accum_t       m_xlat_acc[XLAT_B_NUM];
  unsigned long long m_xlat_dropped = 0;  // records abandoned without completing

  // Little's Law check (N = X*W): N is m_xlat_rec.size(), sampled once per cycle.
  unsigned long long m_xlat_cycles = 0;
  unsigned long long m_xlat_inflight_sum = 0;
  unsigned           m_xlat_inflight_max = 0;

  void xlat_close(uint64_t rid, xlat_req_rec_t &rec, uint64_t cur);

  
  uint64_t vpn(uint64_t va)      const { return va >> m_page_shift; }
  unsigned set_idx(uint64_t vpn) const { return vpn % m_num_sets; }
  static uint64_t req_key(unsigned warp_id, uint64_t vpn) {
    return (vpn << kReqKeyWarpBits) | (warp_id & ((1u << kReqKeyWarpBits) - 1));
  }
  int find_way(unsigned sidx, uint64_t vpn) const;
  int lru_victim(unsigned sidx) const;
  // On a false return (no state change), *in_flight_out receives probe(vpn).
  bool resolve_lookup(const lookup_slot &s, uint64_t cur, bool *in_flight_out = NULL);

  // next_poll: L2 probe-gate expiry (0 = poll now). MSHR_FULL snapshot: wait_merge = merge cap;
  // wait_epoch = ep_consume() (merge cap) or 1 (table full), 0 = none; wait_ep2 = ep_set_idx(set);
  // wait_ep3 = the L2 set index. See gpu_l2_tlb::table_full().
  struct l1_tlb_req { uint64_t vpn; unsigned warp_id; uint64_t enqueue_cycle;
                      uint64_t next_poll; uint64_t wait_epoch; bool wait_merge;
                      uint64_t wait_ep2; uint64_t wait_ep3; };
  std::deque<l1_tlb_req> m_l2_req_queue;   // L1 misses waiting for the L2 TLB (like a cache miss queue)
  std::deque<unsigned>   m_hit_wakeup;     // parked warps resolved to HIT, oldest first

 public:
  // Iterators over the L2 request queue: the drain issues past entries the L2 is still probing.
  typedef std::deque<l1_tlb_req>::iterator l2_req_iter;
  l2_req_iter l2_req_begin() { return m_l2_req_queue.begin(); }
  l2_req_iter l2_req_end()   { return m_l2_req_queue.end(); }
  l2_req_iter l2_req_erase(l2_req_iter it) { return m_l2_req_queue.erase(it); }
  // Walk skip: set when a full cycle_b walk of m_l2_req_queue left every entry asleep. While
  // nothing it saw moves, the next walk would be the same, so its counts are charged in bulk.
  struct walk_memo_t {
    bool armed = false;
    unsigned n_gated = 0, n_merge = 0, n_table = 0;
    uint64_t min_next_poll = 0;            // over the gated entries
    uint64_t ep_set_any = 0, ep_consume = 0;  // L2 snapshots at arm time
    uint64_t enq_seq = 0;                  // m_l2_req_enq at arm time
  };
  walk_memo_t &walk_memo() { return m_walk_memo; }
  uint64_t l2_req_enq_seq() const { return m_l2_req_enq; }
 private:
  walk_memo_t m_walk_memo;
  uint64_t m_l2_req_enq = 0;               // counts every m_l2_req_queue push_back (walk skip)
 public:
};

// L2 TLB MSHR entry: clusters waiting on a VPN, and the PTE each takes via query_completed().
struct l2_mshr_entry_t {
  std::vector<unsigned> pending_clusters;  // cluster_ids (tpc) awaiting this VPN
  bool      pte_ready;  // set true by fill() when GMMU walk completes
  gpu_pte_t pte;        // result pte, valid when pte_ready == true
};

// L2 TLB: per chiplet, shared by its SMs, set-associative, LRU; the MSHR merges walks per VPN.
class gpu_l2_tlb {
 public:
  // Sizes are the -gpgpu_l2_tlb_* options; chiplet_id is used only in logs.
  gpu_l2_tlb(unsigned num_sets, unsigned assoc,
             unsigned mshr_entries, unsigned mshr_max_merge,
             unsigned lookup_latency, unsigned chiplet_id,
             unsigned page_shift = GPU_PAGE_SHIFT);

  // HIT: out_pte filled. MISS: new MSHR entry, the caller starts a walk. MSHR_HIT: merged
  // into an in-flight walk. MSHR_FULL: table full or merge cap, nothing changed, retry.
  enum result_t { HIT, MISS, MSHR_HIT, MSHR_FULL, LOOKUP_PENDING, PORT_BUSY };

  // ready_out: gate expiry for a LOOKUP_PENDING answer; was_probing_out: is_probing() at entry.
  result_t access(uint64_t va, unsigned cluster_id,
                  gpu_pte_t& out_pte, uint64_t current_cycle,
                  uint64_t* ready_out = NULL, bool* was_probing_out = NULL);

  // Latency breakdown: stamp t_pwq_enq; take (and drop) the L2 record at fill time.
  void xlat_mark_pwq_enq(unsigned cluster_id, uint64_t vpn_val, uint64_t cur);
  bool xlat_take_seg(unsigned cluster_id, uint64_t vpn_val, xlat_down_t& out);

  // Bumped by fill(); SMs skip their completion poll while it stands still.
  uint64_t fill_seq() const { return m_fill_seq; }

  // Walk done (gmmu::complete_walk()): insert pte, mark the MSHR entry ready, stamp `walk` stats.
  void fill(uint64_t vpn, const gpu_pte_t& pte, uint64_t current_cycle,
            const struct xlat_walk_t* walk = NULL);

  // Walk done and cluster_id waiting: fill out_pte, remove the waiter (entry freed when empty).
  bool query_completed(uint64_t vpn, unsigned cluster_id, gpu_pte_t& out_pte);
  // VPNs where query_completed(vpn, cluster_id) is true now (may be stale; the poller filters).
  std::vector<uint64_t> &ready_for(unsigned cluster_id) {
    if (cluster_id >= m_ready_by_cluster.size())
      m_ready_by_cluster.resize(cluster_id + 1);
    return m_ready_by_cluster[cluster_id];
  }
  // Side-effect-free query_completed() condition, for the debug check build (P4_CHECK).
  bool completable(uint64_t vpn_val, unsigned cluster_id) const;

  // Request keys hold this many cluster-id bits (checked against the cluster count at startup).
  static const unsigned kReqKeyClusterBits = 12;
  void set_ports(unsigned p) { m_n_ports = p; }
  void set_starts_per_cycle(unsigned n) { m_starts_per_cycle = n; }
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
  // Removes cluster_id from the MSHR entry for vpn (verification harness only).
  void remove_from_mshr(uint64_t vpn, unsigned cluster_id);
  // Deadlock dump: L2 MSHR occupancy and the reason it rejects.
  void xlatdbg_dump(FILE* fp, unsigned chiplet, const char* tag) const;
  // Deadlock dump: non-mutating probe -- what would access() answer for this vpn?
  void xlatdbg_probe(FILE* fp, uint64_t vpn, unsigned cluster_id, uint64_t cur) const;

  void print_stats(FILE* f = stdout) const;
  // -gpgpu_l2_tlb_set_hash: must be called before the first access (no entries yet).
  void set_set_hash(unsigned mode);

 private:
  unsigned m_num_sets;
  unsigned m_assoc;
  unsigned m_chiplet_id;
  unsigned m_page_shift;
  // Set index function (-gpgpu_l2_tlb_set_hash): 0 = vpn % sets, 1 = XOR-fold (see set_idx()).
  unsigned m_set_hash = 0;
  unsigned m_set_bits = 0;   // log2(m_num_sets), valid when m_num_sets is a power of two
  // Per-set usage (ACCELSIM_TLB_SETSTAT=1 prints them): verdicts and array fills per set.
  std::vector<unsigned long long> m_set_lookups;
  std::vector<unsigned long long> m_set_fills;
  // Lookup ports, held until a verdict other than MSHR_FULL; 0 = unlimited.
  unsigned m_n_ports = 0;
  std::unordered_set<uint64_t> m_probing;      // rids holding a port
  unsigned long long m_port_busy_stalls = 0;
  unsigned m_concurrent_max = 0;
  // -gpgpu_l2_tlb_starts_per_cycle: lookups started per cycle (pipelined); 0 = unlimited.
  unsigned m_starts_per_cycle = 0;
  uint64_t m_start_cycle = ~0ull;
  unsigned m_starts_this_cycle = 0;
  unsigned long long m_start_busy = 0;   // requests told to retry next cycle
  unsigned m_mshr_entries;
  unsigned m_mshr_max_merge;
  unsigned m_lookup_latency;

  std::vector<std::vector<l1_tlb_entry_t>> m_sets;  // reuse L1 entry type
  std::unordered_map<uint64_t, l2_mshr_entry_t> m_mshr;
  // See ready_for(). Indexed by global cluster id; each list is a set.
  std::vector<std::vector<uint64_t>> m_ready_by_cluster;
  void ready_add(unsigned cluster_id, uint64_t vpn_val) {
    std::vector<uint64_t> &v = ready_for(cluster_id);
    for (size_t k = 0; k < v.size(); k++)
      if (v[k] == vpn_val) return;
    v.push_back(vpn_val);
  }
  void ready_del(unsigned cluster_id, uint64_t vpn_val) {
    std::vector<uint64_t> &v = ready_for(cluster_id);
    for (size_t k = 0; k < v.size(); k++)
      if (v[k] == vpn_val) {
        v[k] = v.back();
        v.pop_back();
        return;
      }
  }
  // Probe gate per request (rid): ready = cycle the probe resolves; blocked = MSHR-full start
  // already recorded. Keys == m_probing.
  struct gate_t { uint64_t ready; bool blocked; };
  std::unordered_map<uint64_t, gate_t> m_lookup_ready;

  unsigned long long m_hits;
  unsigned long long m_misses;
  unsigned long long m_mshr_hits;
  // Blocked cycles by cause, as in gpu_l1_tlb: merge_full + entries_full == resv_fail_cycles.
  unsigned long long m_mshr_merge_full   = 0;  // cycles blocked by the merge cap (pending_clusters >= max_merge)
  unsigned long long m_mshr_entries_full = 0;  // cycles blocked by a full MSHR table (new VPN)
  // Deadlock dump: which gate branch each access() took.
  unsigned long long m_gate_arm_new = 0;   // branch (3): gate absent -> arm cur+latency
  unsigned long long m_gate_pending = 0;   // branch (1): still inside the window
 public:
  // Charges the LOOKUP_PENDING count of an l2_tlb_request() call the drain skipped.
  void note_gate_pending() { m_gate_pending++; }
  void note_gate_pending_n(unsigned n) { m_gate_pending += n; }  // batch form (walk skip)
  // MSHR_FULL waiters skip re-polls while their verdict cannot change:
  //   table full: while table_full() and the set's epoch (fills, MSHR inserts) stand still;
  //   merge cap: while ep_consume() and the set's epoch stand still.
  bool table_full() const { return m_mshr.size() >= m_mshr_entries; }
  uint64_t ep_consume() const { return m_ep_consume; }
  unsigned set_of(uint64_t vpn_val) const { return set_idx(vpn_val); }
  uint64_t ep_set_idx(unsigned sidx) const { return m_ep_set[sidx]; }
  // Debug check build (P4_CHECK): would resolve() answer MSHR_FULL now? Read-only.
  bool full_verdict(uint64_t vpn_val, bool *merge) const;
  bool last_full_merge() const { return m_last_full_merge; }
  // A skipped re-poll charges what a real retry would: gate_expire and the fail-type counter.
  void note_mshr_full_skip(bool merge) {
    m_gate_expire++;
    if (merge) m_mshr_merge_full++; else m_mshr_entries_full++;
  }
  void note_mshr_full_skip_n(unsigned n_merge, unsigned n_table) {  // batch form (walk skip)
    m_gate_expire += n_merge + n_table;
    m_mshr_merge_full += n_merge;
    m_mshr_entries_full += n_table;
  }
  // Walk skip: moves whenever any m_ep_set[] entry moves (fill or MSHR insert).
  uint64_t ep_set_any() const { return m_ep_set_any; }
 private:
  uint64_t m_ep_consume = 1;         // bumped when a pending_clusters list shrinks
  std::vector<uint64_t> m_ep_set;    // per set, bumped by fills + MSHR inserts
  uint64_t m_ep_set_any = 1;         // see ep_set_any()
  uint64_t m_fill_seq = 0;   // see fill_seq()
  bool m_last_full_merge = false;

  unsigned long long m_gate_expire  = 0;   // branch (1): window elapsed -> resolve()
  unsigned long long m_resv_fail_cycles = 0;
  std::unordered_map<uint64_t, uint64_t> m_resv_fail_start;   // rid -> first fail cycle

  // Latency breakdown: L2 boundaries per in-flight (cluster, vpn); erased when the fill takes it.
  std::unordered_map<uint64_t, xlat_down_t> m_xlat_seg;


  uint64_t vpn(uint64_t va)      const { return va >> m_page_shift; }
  // Set index. Mode 0: vpn % sets. Mode 1: XOR of the VPN's log2(sets)-bit chunks, which
  // spreads power-of-two page strides over all sets (as on NVIDIA TLBs).
  unsigned set_idx(uint64_t vpn) const {
    if (m_set_hash == 0) return vpn % m_num_sets;
    unsigned s = 0;
    for (uint64_t v = vpn; v != 0; v >>= m_set_bits) s ^= (unsigned)(v & (m_num_sets - 1));
    return s;
  }
  static uint64_t req_key(unsigned cluster_id, uint64_t vpn) {
    return (vpn << kReqKeyClusterBits) |
           (cluster_id & ((1u << kReqKeyClusterBits) - 1));
  }
  result_t resolve(uint64_t vpn_val, unsigned sidx, unsigned cluster_id,  // add
                   gpu_pte_t& out_pte, uint64_t cur);
  int find_way(unsigned sidx, uint64_t vpn) const;
  int lru_victim(unsigned sidx) const;
};
