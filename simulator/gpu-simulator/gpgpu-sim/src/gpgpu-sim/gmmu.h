#pragma once
#include <cstdint>
#include <vector>
#include <stdio.h>
#include <queue>
#include <unordered_map>
#include <map>       // MEMLAT_ (temp): ordered sub-partition histogram
#include <utility>   // NEW: for std::pair
#include "xlat_stat.h"  // XLATLAT: xlat_walk_t

//#define ADDR_XLAT_VERIFY
// #define LOG_GMMU

// Forward declaration: gmmu.h and gpu-module.h mutually depend on each other.
class gpu_page_table;
struct va_alloc_t;
class gpu_module;    // NEW: forward declaration
class mem_fetch;     // NEW: forward declaration
class gpu_l2_tlb;

// PWC entry: caches the PA of a next-level PT node, keyed by (level, VA prefix).
// Level 0: PML4[i] -> PDP node PA (key = pml4_idx)
// Level 1: PDP[j]  -> PD  node PA (key = pml4_idx<<9 | pdp_idx)
// Level 2: PD[k]   -> PT  node PA (key = pml4_idx<<18 | pdp_idx<<9 | pd_idx)
// (Leaf level is handled by TLB, not PWC.)
// Entries from all 3 intermediate levels share one unified pool, so each entry
// carries its level to disambiguate keys that collide numerically across levels.
struct pwc_entry_t {
  bool     valid;
  int      level;     // intermediate PT level this entry belongs to (0, 1, 2)
  uint64_t va_key;    // VA-prefix hash (see gmmu::pwc_key())
  uint64_t next_pa;   // PA of the next-level PT node
  uint64_t lru_time;  // cycle timestamp for LRU eviction
};

struct pwq_entry_t {
  uint64_t va;
  unsigned cluster_id;
  uint64_t enqueue_cycle;
};

// XLATLAT: xlat_walk_t lives in xlat_stat.h so gpu_tlb.h can see it too.

// Represents a PTW step ready to issue on the next cycle.
// next_table_pa is guaranteed valid when pushed: ensure_node() or PWC provides it.
struct pending_step_t {
  mem_fetch* parent_mf;
  int        next_level;
  uint64_t   next_table_pa;
};

class gmmu {
 public:
  // chiplet_id: the chiplet this GMMU belongs to (used for logging and Phase 6 PTW routing).
  // pwc_entries: total fully-associative PWC size, unified across all 3 intermediate levels.
  // mod: back-pointer to gpu_module for page table walk and page allocation.
  // NEW: added gpu_module* mod and memory_config* mem_config to constructor
  gmmu(unsigned chiplet_id, unsigned n_ptw, unsigned pwq_depth,
       unsigned pwc_entries,
       gpu_page_table* pt,
       const std::vector<unsigned>* cluster_to_chiplet_map,
       const std::vector<va_alloc_t>* va_alloc_list,
       gpu_module* mod);                                      // NEW parameter 

  // Enqueue a translation request into the PWQ.
  // Returns true if enqueued successfully, false if PWQ is full (caller must stall).
  bool translate(uint64_t va, unsigned cluster_id, uint64_t cycle);

  // Called every L2 clock cycle.
  // Dequeues up to m_n_ptw requests from PWQ and starts asynchronous PTWs
  // (child mem_fetch through icnt/DRAM). Completion is handled later in
  // done_tlb_req() -> complete_walk(), which fills the L2 TLB.
  void cycle(uint64_t current_cycle);

  // Called from gpu_module::icnt_pop when a TLB_FILL_REPLY arrives at this
  // chiplet. Advances the PTW to the next level or completes the walk.
  void done_tlb_req(mem_fetch* child_mf, uint64_t current_cycle);

  bool pwq_full() const { return m_pwq.size() >= m_pwq_max_depth; }

  void print_stats(FILE* f = stdout) const;

  // XLATDBG: outstanding page walks and queue depths at deadlock time.

  void xlatdbg_dump_pending(FILE* fp) const;

  // Set by gpu_module::init() after both GMMU and L2 TLBs are created.
  // One TLB per page size; complete_walk() fills the one matching the VA.
  void set_l2_tlb_4k (gpu_l2_tlb* l2tlb) { m_l2_tlb_4k  = l2tlb; }
  void set_l2_tlb_64k(gpu_l2_tlb* l2tlb) { m_l2_tlb_64k = l2tlb; }
  void set_l2_tlb_2m (gpu_l2_tlb* l2tlb) { m_l2_tlb_2m  = l2tlb; }

  #ifdef ADDR_XLAT_VERIFY
    // Verification-only thin shims exposing private PWC primitives (pure forward).
    uint64_t xlat_pwc_key(uint64_t va, int level) const { return pwc_key(va, level); }
    bool     xlat_pwc_lookup(uint64_t va, int level, uint64_t& out) const {
      return pwc_lookup(va, level, out);
    }
    void     xlat_pwc_insert(uint64_t va, int level, uint64_t pa, uint64_t cyc) {
      pwc_insert(va, level, pa, cyc);
    }
    unsigned xlat_pwc_capacity() const { return m_pwc_entries; }
  #endif
  

 private:
  unsigned    m_chiplet_id;
  unsigned    m_n_ptw;
  unsigned    m_pwq_max_depth;
  unsigned    m_pwc_entries;
  gpu_page_table*                m_pt;
  const std::vector<unsigned>*   m_cluster_to_chiplet_map;
  const std::vector<va_alloc_t>* m_va_alloc_list;
  gpu_module*                    m_mod;    // NEW: back-pointer to gpu_module
  gpu_l2_tlb* m_l2_tlb_4k  = nullptr;  // back-pointer to chiplet L2 TLB (4KB)
  gpu_l2_tlb* m_l2_tlb_64k = nullptr;  // back-pointer to chiplet L2 TLB (64KB)
  gpu_l2_tlb* m_l2_tlb_2m  = nullptr;  // back-pointer to chiplet L2 TLB (2MB)


  // Unified fully-associative PWC shared by levels 0, 1, 2 (not level 3/leaf).
  // Entries are tagged with their level; capacity is m_pwc_entries total.
  std::vector<pwc_entry_t> m_pwc;

  std::queue<pwq_entry_t>                 m_pwq;

  // NEW: tracks phantom parent_mf objects for in-flight walks (P3-1)
  // key: parent_mf pointer (unique per walk); value: current walk level
  std::unordered_map<mem_fetch*, int>  m_active_walks;

  // NEW: walks that are ready for the next PT level but waiting for icnt buffer (P3-1)
  // or received a PWC hit (no DRAM needed, proceed immediately next cycle).
  // pair: {phantom parent_mf, next level to walk}
  std::queue<pending_step_t>  m_pending_next_steps;

  // Statistics
  unsigned long long m_translate_requests;   // total translate() calls
  unsigned long long m_fault_count;

  // per-walk record: keyed by phantom parent_mf pointer.
  // inserted at dequeue time, erased at complete_walk(). Carries the PTW start
  // cycle that the latency stats already used, plus the XLATLAT scalars.
  std::unordered_map<mem_fetch*, xlat_walk_t> m_walk_start_cycles;

  // PTW processing time: time from walk start (dequeue) to complete_walk()
  unsigned long long m_ptw_time_sum;
  unsigned long long m_ptw_complete_count;

  // PTW occupancy: how many PTWs were active per cycle
  unsigned long long m_ptw_active_sum;       // sum of active PTWs each cycle
  unsigned long long m_ptw_full_cycles;      // cycles where active == m_n_ptw
  unsigned long long m_total_cycles;         // total gmmu::cycle() invocations
  uint64_t           m_last_cycle;

  // PWQ queuing delay: cycle count from enqueue to PTW assignment
  unsigned long long m_pwq_queuing_delay_sum;
  unsigned long long m_pwq_dequeue_count;
  unsigned long long m_pwq_max_depth_seen;
  
  // XLATLAT: PWQ inter-arrival distribution. The model assumes arrivals are
  // Poisson (the `M` in M/G/32/FCFS/128), which would make these intervals
  // exponential. Nothing has ever checked that, and GPU access bursts make it
  // doubtful, so the shape is collected and reported as-is.
  uint64_t           m_xlat_last_arrival = 0;
  unsigned long long m_xlat_hist_ia[XLAT_HIST_BINS] = {};

  unsigned long long m_local_steps;    // NEW: child_mf sent to local chiplet DRAM
  unsigned long long m_remote_steps;   // NEW: child_mf sent to remote chiplet DRAM
  unsigned long long m_pwc_hits;       // NEW: levels skipped due to PWC hit
  // Per-level PWC accounting (levels 0..2 = PML4/PDP/PD; the leaf is never cached).
  // A 4 KB walk probes every intermediate level top-down, so lookups[k] equals the
  // number of walks that reached level k; hits sum to m_pwc_hits.
  unsigned long long m_pwc_lookups_lvl[3] = {0, 0, 0};
  unsigned long long m_pwc_hits_lvl[3]    = {0, 0, 0};
  // MEMLAT_ (temp): PT-read step target sub-partition histograms (all / remote-only)
  std::map<unsigned, unsigned long long> m_memlat_subpart_hist;
  std::map<unsigned, unsigned long long> m_memlat_remote_subpart_hist;
  // MEMLAT_ (temp): per-DRAM-step latency split (issue->mem-reply-out, mem-out->gmmu)
  unsigned long long m_memlat_seg_reqmem_sum = 0;
  unsigned long long m_memlat_seg_reply_sum  = 0;
  unsigned long long m_memlat_seg_cnt        = 0;
  // FUNNEL_ (temp): single-cluster (repr_cluster) PTW endpoint funnel evidence.
  //   req side  : all PTW requests inject through repr_cluster's ONE icnt input
  //               port -> req_full counts requeues when that input buffer is full.
  //   reply side: all PTW replies converge on repr_cluster's ONE icnt output port
  //               -> seg_reply_max is the tail of that convergence queueing.
  unsigned long long m_funnel_req_ok   = 0;   // PTW requests successfully injected
  unsigned long long m_funnel_req_full = 0;   // PTW requests requeued (icnt input full)
  unsigned long long m_memlat_seg_reply_max = 0;
  // FUNNEL_ (temp): split reply seg B (=memout->gmmu) using dest-cluster out_buffer
  // arrival (mf->icnt_receive_time). transit = t1->arrival (xbar+ring), lasthop =
  // arrival->gmmu pop (wait in dest cluster egress FIFO behind normal SM traffic).
  unsigned long long m_memlat_seg_transit_sum = 0;
  unsigned long long m_memlat_seg_lasthop_sum = 0;
  unsigned long long m_memlat_seg_lasthop_max = 0;
  unsigned long long m_memlat_seg_split_cnt   = 0;

  // Fix A: PTW icnt endpoint round-robin cursor. PTW req/reply borrow a cluster's
  // icnt port; rotating across this chiplet's clusters spreads the single-port
  // funnel (see docs/ptw_endpoint_funnel_redesign.ko.md).
  unsigned m_rr_endpoint = 0;

  // NEW: issue one child_mf for PT level `level` using table base PA `table_pa`.
  // If PWC hit: does NOT issue DRAM request; instead pushes to m_pending_next_steps.
  void pt_walk(mem_fetch* parent_mf, int level, uint64_t table_pa,
               uint64_t current_cycle);

  // Finish a walk: record PTW latency, fill the L2 TLB with the result PTE,
  // then free the phantom parent_mf and drop it from m_active_walks.
  void complete_walk(mem_fetch* parent_mf, uint64_t current_cycle);

  uint64_t pwc_key(uint64_t va, int level) const;
  bool pwc_lookup(uint64_t va, int level, uint64_t& out_next_pa) const;
  void pwc_insert(uint64_t va, int level, uint64_t next_pa, uint64_t cycle);
};