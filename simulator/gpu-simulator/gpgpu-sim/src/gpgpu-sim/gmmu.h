#pragma once
#include <cstdint>
#include <vector>
#include <stdio.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <map>       // diagnostics: ordered sub-partition histogram
#include "xlat_stat.h"  // xlat_walk_t (per-walk latency record)

//#define ADDR_XLAT_VERIFY
// #define LOG_GMMU

// Forward declaration: gmmu.h and gpu-module.h mutually depend on each other.
class gpu_page_table;
struct va_alloc_t;
class gpu_module;
class mem_fetch;
class gpu_l2_tlb;

// PWC entry: PA of the next-level PT node for one (level, VA prefix). Leaf PTEs are not cached.
// All levels share one pool, so a lookup matches both level and key.
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

// xlat_walk_t lives in xlat_stat.h so gpu_tlb.h can use it too.

// A walk step to run in the next cycle(); next_table_pa is always valid when pushed.
struct pending_step_t {
  mem_fetch* parent_mf;
  int        next_level;
  uint64_t   next_table_pa;
};

class gmmu {
 public:
  // chiplet_id: this GMMU's chiplet. pwc_entries: total entries of the PWC shared by levels 0-2.
  gmmu(unsigned chiplet_id, unsigned n_ptw, unsigned pwq_depth,
       unsigned pwc_entries,
       gpu_page_table* pt,
       const std::vector<unsigned>* cluster_to_chiplet_map,
       const std::vector<va_alloc_t>* va_alloc_list,
       gpu_module* mod);

  // Enqueue a walk request into the PWQ. Returns false if the PWQ is full.
  bool translate(uint64_t va, unsigned cluster_id, uint64_t cycle);

  // Per L2 cycle: run ready walk steps, then start PWQ walks while fewer than m_n_ptw are active.
  void cycle(uint64_t current_cycle);

  // A PT read reply arrived: advance the walk to the next level or complete it.
  void done_tlb_req(mem_fetch* child_mf, uint64_t current_cycle);

  bool pwq_full() const { return m_pwq.size() >= m_pwq_max_depth; }

  void print_stats(FILE* f = stdout) const;

  // Deadlock dump: outstanding page walks and queue depths.

  void xlatdbg_dump_pending(FILE* fp) const;

  // L2 TLBs, one per page size; complete_walk() fills the one matching the VA.
  void set_l2_tlb_4k (gpu_l2_tlb* l2tlb) { m_l2_tlb_4k  = l2tlb; }
  void set_l2_tlb_64k(gpu_l2_tlb* l2tlb) { m_l2_tlb_64k = l2tlb; }
  void set_l2_tlb_2m (gpu_l2_tlb* l2tlb) { m_l2_tlb_2m  = l2tlb; }

  #ifdef ADDR_XLAT_VERIFY
    // Verification-only wrappers around the private PWC functions.
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
  gpu_module*                    m_mod;    // back-pointer to gpu_module
  gpu_l2_tlb* m_l2_tlb_4k  = nullptr;  // back-pointer to chiplet L2 TLB (4KB)
  gpu_l2_tlb* m_l2_tlb_64k = nullptr;  // back-pointer to chiplet L2 TLB (64KB)
  gpu_l2_tlb* m_l2_tlb_2m  = nullptr;  // back-pointer to chiplet L2 TLB (2MB)


  // Fully-associative PWC shared by levels 0-2; m_pwc_entries entries in total.
  std::vector<pwc_entry_t> m_pwc;

  std::queue<pwq_entry_t>                 m_pwq;

  // Walks in flight, by phantom parent_mf; the count is the number of busy walkers.
  std::unordered_set<mem_fetch*>  m_active_walks;

  // Steps to run next cycle(): after a PWC hit or a PT reply, or a retry when ICNT was full.
  std::queue<pending_step_t>  m_pending_next_steps;

  // Statistics
  unsigned long long m_translate_requests;   // total translate() calls
  unsigned long long m_fault_count;

  // Per-walk record (start cycle, latency breakdown), from PWQ dequeue to complete_walk().
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
  
  // PWQ inter-arrival histogram, to check the Poisson-arrival assumption of the queue model.
  uint64_t           m_xlat_last_arrival = 0;
  unsigned long long m_xlat_hist_ia[XLAT_HIST_BINS] = {};

  unsigned long long m_local_steps;    // child_mf sent to local chiplet DRAM
  unsigned long long m_remote_steps;   // child_mf sent to remote chiplet DRAM
  unsigned long long m_pwc_hits;       // levels skipped due to PWC hit
  // Per-level PWC lookups and hits (0-2 = PML4/PDP/PD); hits sum to m_pwc_hits.
  unsigned long long m_pwc_lookups_lvl[3] = {0, 0, 0};
  unsigned long long m_pwc_hits_lvl[3]    = {0, 0, 0};
  // Diagnostics (ACCELSIM_DIAG): PT-read target sub-partition histograms (all / remote-only)
  std::map<unsigned, unsigned long long> m_memlat_subpart_hist;
  std::map<unsigned, unsigned long long> m_memlat_remote_subpart_hist;
  // Diagnostics (ACCELSIM_DIAG): per-DRAM-step latency split (issue->mem-reply-out, mem-out->gmmu)
  unsigned long long m_memlat_seg_reqmem_sum = 0;
  unsigned long long m_memlat_seg_reply_sum  = 0;
  unsigned long long m_memlat_seg_cnt        = 0;
  // Diagnostics (ACCELSIM_DIAG): PTW request injections, ICNT-full requeues, max reply latency.
  unsigned long long m_funnel_req_ok   = 0;   // PTW requests successfully injected
  unsigned long long m_funnel_req_full = 0;   // PTW requests requeued (icnt input full)
  unsigned long long m_memlat_seg_reply_max = 0;
  // Diagnostics (ACCELSIM_DIAG): reply time split at the dest-cluster arrival:
  // transit = mem out -> arrival (xbar + ring), lasthop = arrival -> GMMU pop.
  unsigned long long m_memlat_seg_transit_sum = 0;
  unsigned long long m_memlat_seg_lasthop_sum = 0;
  unsigned long long m_memlat_seg_lasthop_max = 0;
  unsigned long long m_memlat_seg_split_cnt   = 0;

  // PTWs borrow a cluster's ICNT port, rotated over this chiplet's clusters to spread traffic.
  unsigned m_rr_endpoint = 0;

  // One walk step: PWC hit -> queue the next level; else send a PT read, or retry it next cycle.
  void pt_walk(mem_fetch* parent_mf, int level, uint64_t table_pa,
               uint64_t current_cycle);

  // Finish a walk: record latency, fill the L2 TLB, free the phantom parent_mf.
  void complete_walk(mem_fetch* parent_mf, uint64_t current_cycle);

  uint64_t pwc_key(uint64_t va, int level) const;
  bool pwc_lookup(uint64_t va, int level, uint64_t& out_next_pa) const;
  // Like pwc_lookup(), but a hit refreshes lru_time (LRU on the walk path).
  bool pwc_lookup_touch(uint64_t va, int level, uint64_t& out_next_pa, uint64_t cycle);
  void pwc_insert(uint64_t va, int level, uint64_t next_pa, uint64_t cycle);
};