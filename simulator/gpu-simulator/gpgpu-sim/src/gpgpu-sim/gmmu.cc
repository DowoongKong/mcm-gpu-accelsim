#include <cstdlib>   // getenv (verification log gate)
#include "gmmu.h"
#include "gpu-module.h"  // gpu_page_table, va_alloc_t, gpu_pte_t, GPU_PAGE_SHIFT, GPU_PAGE_SIZE
#include "mem_fetch.h"  // NEW: for child_mf creation (P3-1)
#include "shader.h"     // NEW: for READ_PACKET_SIZE (P3-1)
#include "gpu-sim.h"  // full definition of memory_config (needed for gpgpu_ctx member access)
#include "icnt_wrapper.h"  // icnt_has_buffer declaration
#include "gpu_tlb.h"   // gpu_l2_tlb full definition for l2_tlb->fill()

// Verification logging: per-walk event lines, OFF unless ACCELSIM_LOG_GMMU is set.
// Runtime-gated so one binary serves production (no output, digest-neutral)
// and verification runs (recompute walks / local-remote steps / occupancy
// from the log and cross-check the GMMU_STAT counters).

// 44-2: pure consistency checks on hot paths. Compiled in only with
// -DXLAT_DEBUG_CHECKS; the default build skips them (the checks re-walk the
// page table / re-hash the MSHR and cost real time at high walk rates).
#ifdef XLAT_DEBUG_CHECKS
#define XLAT_HOT_ASSERT(x) assert(x)
#else
#define XLAT_HOT_ASSERT(x) ((void)0)
#endif

static bool accelsim_log_gmmu() {
  static const bool on = (getenv("ACCELSIM_LOG_GMMU") != NULL);
  return on;
}


gmmu::gmmu(unsigned chiplet_id, unsigned n_ptw, unsigned pwq_depth,
           unsigned pwc_entries,
           gpu_page_table*                pt,
           const std::vector<unsigned>*   cluster_to_chiplet_map,
           const std::vector<va_alloc_t>* va_alloc_list,
           gpu_module*                    mod)  // NEW
    : m_chiplet_id(chiplet_id),
      m_n_ptw(n_ptw),
      m_pwq_max_depth(pwq_depth),
      m_pwc_entries(pwc_entries),
      m_pt(pt),
      m_cluster_to_chiplet_map(cluster_to_chiplet_map),
      m_va_alloc_list(va_alloc_list),
      m_mod(mod),            // NEW
      m_translate_requests(0),
      m_fault_count(0),
      m_ptw_time_sum(0),
      m_ptw_complete_count(0),
      m_ptw_active_sum(0),
      m_ptw_full_cycles(0),
      m_total_cycles(0),
      m_last_cycle(0),
      m_pwq_queuing_delay_sum(0),
      m_pwq_dequeue_count(0),
      m_pwq_max_depth_seen(0),
      m_local_steps(0),   // NEW
      m_remote_steps(0),   // NEW
      m_pwc_hits(0) {   // NEW
  m_pwc.resize(m_pwc_entries, {false, -1, 0ULL, 0ULL, 0ULL});
}

bool gmmu::translate(uint64_t va, unsigned cluster_id, uint64_t cycle) {
  if (pwq_full()) return false;  // caller (SM) must stall and retry

  // XLATLAT: measured on ACCEPTED arrivals only. A PWQ-full rejection is not an
  // arrival to the queueing system -- counting it would fabricate a burst that
  // the model never sees.
  if (m_xlat_last_arrival != 0 && cycle >= m_xlat_last_arrival)
    m_xlat_hist_ia[xlat_hist_bin(cycle - m_xlat_last_arrival)]++;
  m_xlat_last_arrival = cycle;

  m_pwq.push({va, cluster_id, cycle});
  m_translate_requests++;
  // owner = first accessing chiplet. Routed: reference mode defers the owner
  // insert to the sequential commit point (A1); other modes insert immediately.
  m_pt->note_first_touch(va, m_chiplet_id);

  uint64_t depth = (uint64_t)m_pwq.size();
  if (depth > m_pwq_max_depth_seen) m_pwq_max_depth_seen = depth;
  return true;  // successfully enqueued
}

void gmmu::cycle(uint64_t current_cycle) {
  m_last_cycle = current_cycle;   // updated here; used by done_tlb_req() (cycle=0 workaround)
  m_total_cycles++;               // counts every cycle() invocation for occupancy denominator

  // 39: idle early-exit. Every branch below is gated on one of these queues;
  // with all of them empty the remainder of cycle() touches no state beyond
  // the two unconditional updates above (occupancy accumulators below add 0).
  // The queues are filled only by translate()/pt_walk()/memory returns, so
  // skipping quiet cycles is exact.
  if (m_pending_next_steps.empty() && m_pwq.empty() && m_active_walks.empty())
    return;

  // --- Step 1: process walks that advanced via PWC hit or awaiting icnt buffer ---
  // Process all pending next-level steps first (PWC hits have no DRAM cost).
  // For icnt-buffer-stalled retries, pt_walk() will re-stall if still full.
  unsigned processed_pending = 0;
  unsigned pending_size = (unsigned)m_pending_next_steps.size();
  while (processed_pending < pending_size && !m_pending_next_steps.empty()) {
    pending_step_t step = m_pending_next_steps.front();
    m_pending_next_steps.pop();
    processed_pending++;

    // next_table_pa is guaranteed valid: ensure_node() was called in
    // done_tlb_req() before this entry was pushed, or it came from a PWC hit.
    assert(step.next_table_pa != (uint64_t)-1);
    pt_walk(step.parent_mf, step.next_level, step.next_table_pa, current_cycle);
  }


  // --- Step 2: dequeue new requests from PWQ and start walks ---
  unsigned active_this_cycle = (unsigned)m_active_walks.size();
  while (active_this_cycle < m_n_ptw && !m_pwq.empty()) {
    pwq_entry_t entry = m_pwq.front();
    m_pwq.pop();

    // -1: enqueue happens in the same L2-domain tick that could not serve it;
    // the first service opportunity is the NEXT tick, so a dequeue there is
    // zero queueing delay, not one.
    uint64_t queuing_delay = (current_cycle > entry.enqueue_cycle)
                             ? current_cycle - entry.enqueue_cycle - 1 : 0;
    m_pwq_queuing_delay_sum += queuing_delay;
    m_pwq_dequeue_count++;

    // Create phantom parent_mf to carry {va, cluster_id} through the walk.
    // This mf is NEVER pushed to icnt; it is a software-only walk token.
    const memory_config* mem_cfg = m_mod->get_memory_config();
    mem_access_t phantom_access(GLOBAL_ACC_R, entry.va, 8, false,
                                mem_cfg->gpgpu_ctx);
    mem_fetch* parent_mf = new mem_fetch(
        phantom_access,
        nullptr,             // no warp instruction
        0,                   // streamID = 0
        READ_PACKET_SIZE,    // ctrl_size
        0,                   // wid = 0 (no real warp)
        entry.cluster_id,    // sid = cluster_id (dummy SM within cluster)
        entry.cluster_id,    // tpc = cluster_id (determines reply routing)
        mem_cfg,
        current_cycle);
    // parent_mf->get_va() == entry.va (constructor sets m_va = access.get_addr())
    // parent_mf->get_tpc() == entry.cluster_id

    m_active_walks[parent_mf] = 0;
    if (accelsim_log_gmmu())
      printf("[GMMU_START] chiplet=%u va=0x%llx cyc=%llu pwq_wait=%llu\n",
             m_chiplet_id, (unsigned long long)entry.va,
             (unsigned long long)m_total_cycles,
             (unsigned long long)queuing_delay);
    // XLATLAT: t6. entry.enqueue_cycle is t5, carried from translate().
    xlat_walk_t &wr = m_walk_start_cycles[parent_mf];
    wr.t_pwq_enq    = entry.enqueue_cycle;
    wr.t_walk_start = current_cycle;
    active_this_cycle++;

    // PML4 is always pre-allocated in gpu_page_table constructor.
    // Demand paging faults are handled level-by-level inside done_tlb_req()
    // as each DRAM reply arrives. No pre-fault check needed here.
    // THIS chiplet's CR3. Without -gmmu_local_ptw every chiplet shares the one
    // root page, so this is the pre-existing behavior.
    uint64_t pml4_pa = m_pt->get_pml4_base_pa(m_chiplet_id);
    pt_walk(parent_mf, 0, pml4_pa, current_cycle);
  }

  #ifdef ADDR_XLAT_VERIFY
    assert(m_active_walks.size() <= m_n_ptw &&
          "V-PTW: active walks must never exceed n_ptw");
    assert(m_active_walks.size() == m_walk_start_cycles.size() &&
          "V-PTW: active_walks and walk_start_cycles stay paired");
  #endif

  // PTW occupancy stats
  m_ptw_active_sum += (unsigned long long)m_active_walks.size();
  if ((unsigned)m_active_walks.size() == m_n_ptw) m_ptw_full_cycles++;
}

// Recursive function to perform a page table walk step at the given level for parent_mf. (P3-1)
void gmmu::pt_walk(mem_fetch* parent_mf, int level, uint64_t table_pa,
                   uint64_t current_cycle) {
  uint64_t va = parent_mf->get_va();
  // Leaf level for this VA's page size: 3 (PT) for 4KB, 2 (PD) for 2MB.
  const int leaf_level = pt_leaf_level(m_mod->page_shift_for_va(va));

  // Check PWC for intermediate levels only; the leaf level is never cached.
  // For 2MB that means levels 0 and 1, since level 2 is the leaf.
  if (level < leaf_level) {
    uint64_t cached_pa;
    if (pwc_lookup(va, level, cached_pa)) {
      m_pwc_hits++;
      {  // XLATLAT: a PWC hit costs no memory read; count it as a skipped level.
        auto w = m_walk_start_cycles.find(parent_mf);
        if (w != m_walk_start_cycles.end()) w->second.n_pwc_hit++;
      }
      // cached_pa is the validated PA of the next-level node from PWC.
      m_pending_next_steps.push({parent_mf, level + 1, cached_pa});
      if (accelsim_log_gmmu()) {
      printf("[GMMU_PWC] chiplet=%u va=0x%llx level=%d -> HIT next_pa=0x%llx\n",
             m_chiplet_id, (unsigned long long)va, level,
             (unsigned long long)cached_pa);
      }
      return;
    }
  }

  // Compute the physical address of the entry at this level within table_pa.
  // Each PT entry is 8 bytes; index into the node by level-specific VA bits. At
  // level 3 the index width follows the page size: 9 bits into the 512-entry 4KB
  // PT, or 5 bits into the 32-entry 64KB big PT.
  unsigned idx;
  switch (level) {
    case 0: idx = (unsigned)pt_pml4_idx(va); break;
    case 1: idx = (unsigned)pt_pdp_idx(va);  break;
    case 2: idx = (unsigned)pt_pd_idx(va);   break;
    default:  // level 3
      idx = (m_mod->page_shift_for_va(va) == GPU_64K_SHIFT)
          ? (unsigned)pt_pt64k_idx(va)
          : (unsigned)pt_pt_idx(va);
      break;
  }
  uint64_t entry_pa = table_pa + (uint64_t)(idx * sizeof(uint64_t));

  // Determine if this DRAM access is local or remote relative to m_chiplet_id.
  unsigned dst_chiplet = m_mod->pfn_to_chiplet_via_addrdec(entry_pa);
  bool is_remote = (dst_chiplet != m_chiplet_id);

  // Build the child mem_fetch that will traverse icnt -> L2 -> DRAM.
  const memory_config* mem_cfg = m_mod->get_memory_config();
  
  // PT entry is 8 bytes, but L2 cache operates at sector granularity (32 bytes).
  // Compute which sector of the 128-byte cache line contains entry_pa.
  unsigned sector_idx = (entry_pa % MAX_MEMORY_ACCESS_SIZE) / SECTOR_SIZE;
  mem_access_sector_mask_t ptw_sector_mask;
  ptw_sector_mask.set(sector_idx);
  mem_access_byte_mask_t ptw_byte_mask;
  for (unsigned b = sector_idx * SECTOR_SIZE;
       b < (sector_idx + 1) * SECTOR_SIZE; b++)
    ptw_byte_mask.set(b);

  mem_access_t ptw_access(GLOBAL_ACC_R, entry_pa, SECTOR_SIZE, /*wr=*/false,
                          active_mask_t(), ptw_byte_mask, ptw_sector_mask,
                          mem_cfg->gpgpu_ctx);


  // tpc of child_mf determines where the reply is routed back.
  // Fix A: a PTW borrows a cluster's icnt endpoint. Rotating round-robin across
  // this chiplet's clusters spreads the single-port funnel (all PTW req/reply
  // used to converge on the first cluster). Any cluster of this chiplet routes
  // the reply to the same GMMU (gpu_module::icnt_pop -> cluster_to_chiplet), and
  // the target DRAM is decided by the PT-node PA (get_sub_partition_id), not by
  // sid/tpc, so spreading changes only where the reply returns.
  // See docs/ptw_endpoint_funnel_redesign.ko.md.
  unsigned cl_begin = m_mod->get_chiplet(m_chiplet_id)->get_cluster_global_begin();
  unsigned cl_count = m_mod->get_chiplet(m_chiplet_id)->get_num_clusters();
  unsigned endpoint = cl_begin + (m_rr_endpoint++ % cl_count);

  mem_fetch* child_mf = new mem_fetch(
      ptw_access,
      nullptr,           // no warp instruction
      0,                 // streamID = 0
      READ_PACKET_SIZE,  // ctrl_size
      0,                 // wid = 0
      endpoint,          // sid = borrowed cluster (rotated)
      endpoint,          // tpc = borrowed cluster (routes reply back here)
      mem_cfg,
      current_cycle);

  // Override mf_type from READ_REQUEST to TLB_FILL_ACC so the reply can be
  // distinguished from normal SM READ_REPLY in gpu_module::icnt_push.
  child_mf->set_type(TLB_FILL_ACC);
  child_mf->set_tlb_depth(level);
  child_mf->set_parent_mf(parent_mf);

  // Check icnt input buffer availability before pushing.
  unsigned src_dev = m_mod->cluster2device(endpoint);
  unsigned dst_dev = m_mod->mem2device(child_mf->get_sub_partition_id());
  unsigned flit_size = child_mf->get_ctrl_size();

  if (!icnt_has_buffer(src_dev, flit_size)) {
    // icnt input buffer full: retry this step next cycle.
    m_funnel_req_full++;   // FUNNEL_ (temp): request blocked by repr_cluster input port
    {  // XLATLAT: backpressure is not page-table access time -- keep it separate
       // so S7 is not inflated by interconnect congestion. The retry re-enters at
       // the SAME level, so no step is counted twice.
      auto w = m_walk_start_cycles.find(parent_mf);
      if (w != m_walk_start_cycles.end() && w->second.icnt_bp < 0xFFFF)
        w->second.icnt_bp++;
    }
    delete child_mf;
    // Retry at the same level next cycle with the same table_pa (still valid).
    m_pending_next_steps.push({parent_mf, level, table_pa});
    if (accelsim_log_gmmu()) {
    printf("[GMMU_STALL] chiplet=%u va=0x%llx level=%d icnt_full\n",
           m_chiplet_id, (unsigned long long)va, level);
    }
    return;
  }

  if (is_remote) m_remote_steps++;
  else           m_local_steps++;
  {  // XLATLAT: one page-table read actually left for memory. n_remote / n_steps
     // is the per-walk q that E[S] = E[M]*t_local + E[R]*t_icnt needs.
    auto w = m_walk_start_cycles.find(parent_mf);
    if (w != m_walk_start_cycles.end()) {
      w->second.n_steps++;
      if (is_remote) w->second.n_remote++;
    }
  }
  // MEMLAT_ (temp): which memory sub-partition each PT-read step targets, split
  // local/remote. Detects a single-partition hotspot at the page-table owner.
  m_memlat_subpart_hist[child_mf->get_sub_partition_id()] += 1;
  if (is_remote) m_memlat_remote_subpart_hist[child_mf->get_sub_partition_id()] += 1;

  m_funnel_req_ok++;   // FUNNEL_ (temp): request injected through repr_cluster input port
  m_mod->icnt_push(src_dev, dst_dev, child_mf, flit_size);
  if (accelsim_log_gmmu()) {
  printf("[GMMU_PTW] chiplet=%u va=0x%llx level=%d entry_pa=0x%llx %s cyc=%llu\n",
         m_chiplet_id, (unsigned long long)va, level,
         (unsigned long long)entry_pa, is_remote ? "REMOTE" : "LOCAL",
         (unsigned long long)m_total_cycles);
  }
}

// Called from gpu_module::icnt_push when a TLB_FILL_REPLY is intercepted. (P3-1)
void gmmu::done_tlb_req(mem_fetch* child_mf, uint64_t current_cycle) {
  // gpu_module::icnt_push() passes cycle=0; fall back to last known cycle.
  if (current_cycle == 0) current_cycle = m_last_cycle;
  
  int       level     = child_mf->get_tlb_depth();
  mem_fetch* parent_mf = child_mf->get_parent_mf();
  uint64_t  va        = parent_mf->get_va();
  // Leaf level for this VA's page size: 3 (PT) for 4KB, 2 (PD) for 2MB.
  const int leaf_level = pt_leaf_level(m_mod->page_shift_for_va(va));

  // MEMLAT_ (temp): per-DRAM-step latency breakdown. This GMMU's steps are all
  // local (chiplet==owner) or all remote; per-chiplet averages separate them.
  //   seg_reqmem = issue -> mem pushed reply   (req icnt + mem service)
  //   seg_reply  = mem pushed reply -> arrival  (reply icnt back to this GMMU)
  {
    unsigned long long t0 = child_mf->get_timestamp();          // issue
    unsigned long long t1 = child_mf->get_return_timestamp();   // mem reply out
    unsigned long long now = current_cycle;
    if (t1 >= t0 && now >= t1 && t1 != 0) {
      m_memlat_seg_reqmem_sum += (t1 - t0);
      m_memlat_seg_reply_sum  += (now - t1);
      m_memlat_seg_cnt++;
      if ((now - t1) > m_memlat_seg_reply_max)   // FUNNEL_ (temp): reply tail
        m_memlat_seg_reply_max = (now - t1);
      // FUNNEL_ (temp): split B at dest-cluster out_buffer arrival timestamp.
      unsigned long long arr = child_mf->get_icnt_receive_time();
      if (arr != 0 && arr >= t1 && now >= arr) {
        m_memlat_seg_transit_sum += (arr - t1);   // xbar+ring transit
        m_memlat_seg_lasthop_sum += (now - arr);  // wait in dest cluster egress FIFO
        m_memlat_seg_split_cnt++;
        if ((now - arr) > m_memlat_seg_lasthop_max)
          m_memlat_seg_lasthop_max = (now - arr);
      }
    }
  }

  if (level < leaf_level) {
    // The DRAM reply represents reading the PT entry at this level.
    // Check whether the next-level node exists (P=1 in entry means node exists).
    // Address THIS chiplet's copy of the next node. Without -gmmu_local_ptw all
    // copies are the same page, so this is the pre-existing behavior.
    uint64_t next_table_pa = m_pt->get_node_base_pa(va, level + 1, m_chiplet_id);
    if (next_table_pa == (uint64_t)-1) {
      // P=0: the next-level PT node is absent. This is the demand paging fault.
      // Create only the missing node on the requesting cluster's chiplet.
      unsigned chiplet_id = m_pt->first_touch_node_owner(va, level + 1);
      m_pt->ensure_node(va, level + 1, chiplet_id);
      // ensure_node() returns the OWNER's copy; re-query for this chiplet's.
      next_table_pa = m_pt->get_node_base_pa(va, level + 1, m_chiplet_id);
      m_fault_count++;
      if (accelsim_log_gmmu()) {
      printf("[GMMU_FAULT] chiplet=%u va=0x%llx level=%d->%d node absent,"
             " created on chiplet=%u pa=0x%llx\n",
             m_chiplet_id, (unsigned long long)va, level, level + 1,
             chiplet_id, (unsigned long long)next_table_pa);
      }
    }
    // Cache the next-level PA in PWC for future walks to the same VA prefix.
    pwc_insert(va, level, next_table_pa, current_cycle);
    delete child_mf;
    // Include next_table_pa so cycle() Step 1 skips get_node_base_pa re-query.
    m_pending_next_steps.push({parent_mf, level + 1, next_table_pa});
    if (accelsim_log_gmmu()) {
    printf("[GMMU_DONE] chiplet=%u va=0x%llx level=%d -> level=%d"
           " next_pa=0x%llx queued\n",
           m_chiplet_id, (unsigned long long)va, level, level + 1,
           (unsigned long long)next_table_pa);
    }
  } else {
    // Leaf level (3 = PT for 4KB, 2 = PD for 2MB): DRAM returned the leaf entry.
    // If the data page is absent (first access), allocate it now. walk() and
    // map_leaf() are both page-size aware, so no branching is needed here.
    delete child_mf;
    const gpu_pte_t* pte = m_pt->walk(va);
    if (!pte || !pte->present()) {
      unsigned chiplet_id = m_pt->first_touch_data_owner(va);
      m_pt->map_leaf(va, chiplet_id);
      m_fault_count++;
      if (accelsim_log_gmmu()) {
      printf("[GMMU_FAULT] chiplet=%u va=0x%llx level=%d leaf absent,"
             " data page allocated on chiplet=%u\n",
             m_chiplet_id, (unsigned long long)va, leaf_level,
             chiplet_id);
      }
    }
    // 44-2: duplicate of complete_walk()'s assert on the pte it actually
    // uses (line below); as a plain assert this cost two extra page walks.
    XLAT_HOT_ASSERT(m_pt->walk(va) && m_pt->walk(va)->present());
    complete_walk(parent_mf, current_cycle);
  }

}

// Completes a page table walk by storing the final PA result and cleaning up the parent_mf token. (P3-1)
void gmmu::complete_walk(mem_fetch* parent_mf, uint64_t current_cycle) {
  uint64_t va         = parent_mf->get_va();
  unsigned cluster_id = parent_mf->get_tpc();

  const gpu_pte_t* pte = m_pt->walk(va);
  assert(pte && pte->present());

  // PTW latency measurement: dequeue-to-complete.
  xlat_walk_t wr;
  bool have_wr = false;
  auto it = m_walk_start_cycles.find(parent_mf);
  if (it != m_walk_start_cycles.end()) {
    m_ptw_time_sum += current_cycle - it->second.t_walk_start;
    m_ptw_complete_count++;
    wr = it->second;          // XLATLAT: scalars ride up with the fill
    have_wr = true;
    m_walk_start_cycles.erase(it);
  }

  // Deliver PTW result to L2 TLB.
  // fill() inserts pte into L2 TLB array and marks L2 MSHR pte_ready=true.
  // Each waiting SM discovers this via query_completed() in ldst_unit::cycle()
  // and fills its own L1 TLB independently.
  // Fill the L2 TLB that matches this VA's page size, keyed by that size's VPN.
  const unsigned pshift = m_mod->page_shift_for_va(va);
  gpu_l2_tlb* l2 = (pshift == GPU_HUGE_SHIFT) ? m_l2_tlb_2m
                 : (pshift == GPU_64K_SHIFT)  ? m_l2_tlb_64k
                                              : m_l2_tlb_4k;
  assert(l2 != nullptr);
  // XLATLAT: the GMMU cannot address the requesting SM -- translate() replaced the
  // requester's cluster id with the chiplet representative so page placement lands
  // on this chiplet. The L2 MSHR is what knows who is waiting, so the walk scalars
  // are handed to it and it stamps every pending cluster.
  // current_cycle is t7 (walk done), so fill() needs no separate timestamp.
  l2->fill(page_vpn(va, pshift), *pte, current_cycle, have_wr ? &wr : NULL);

  // pfn is always a 4KB frame number; the offset width follows the page size.
  uint64_t pa = (pte->pfn() << GPU_PAGE_SHIFT) | page_offset(va, pshift);
  if (accelsim_log_gmmu()) {
  printf("[GMMU_COMPLETE] chiplet=%u cluster=%u va=0x%llx -> pa=0x%llx cyc=%llu\n",
         m_chiplet_id, cluster_id,
         (unsigned long long)va, (unsigned long long)pa,
         (unsigned long long)m_total_cycles);
  }

  // Remove from active walk tracking and free phantom parent_mf.
  m_active_walks.erase(parent_mf);
  delete parent_mf;
}

uint64_t gmmu::pwc_key(uint64_t va, int level) const {
  uint64_t pml4 = (va >> 39) & 0x1FFULL;
  uint64_t pdp  = (va >> 30) & 0x1FFULL;
  uint64_t pd   = (va >> 21) & 0x1FFULL;
  switch (level) {
    case 0: return pml4;
    case 1: return (pml4 << 9)  | pdp;
    case 2: return (pml4 << 18) | (pdp << 9) | pd;
    default: return (uint64_t)-1;
  }
}

bool gmmu::pwc_lookup(uint64_t va, int level, uint64_t& out_next_pa) const {
  uint64_t key = pwc_key(va, level);
  // Unified pool: match both level and key (keys can collide across levels).
  for (const auto& e : m_pwc) {
    if (e.valid && e.level == level && e.va_key == key) {
      out_next_pa = e.next_pa;
      return true;
    }
  }
  return false;
}

void gmmu::pwc_insert(uint64_t va, int level, uint64_t next_pa, uint64_t cycle) {
  uint64_t key = pwc_key(va, level);
  // Update if already present
  for (auto& e : m_pwc) {
    if (e.valid && e.level == level && e.va_key == key) {
      e.next_pa  = next_pa;
      e.lru_time = cycle;
      return;
    }
  }
  // Fill empty slot first
  for (auto& e : m_pwc) {
    if (!e.valid) {
      e = {true, level, key, next_pa, cycle};
      return;
    }
  }
  // All slots full: evict LRU across the whole unified pool
  pwc_entry_t* lru = &m_pwc[0];
  for (auto& e : m_pwc)
    if (e.lru_time < lru->lru_time) lru = &e;
  *lru = {true, level, key, next_pa, cycle};
}

void gmmu::print_stats(FILE* f) const {
  unsigned long long total_cycles = m_total_cycles > 0 ? m_total_cycles : 1;
  unsigned long long n_deq        = m_pwq_dequeue_count > 0 ? m_pwq_dequeue_count : 1;

  fprintf(f, "[GMMU_STAT] chiplet=%u\n", m_chiplet_id);
  fprintf(f, "  requests         : %llu\n", m_translate_requests);
  fprintf(f, "  faults           : %llu\n", m_fault_count);
  fprintf(f, "  PTW slots        : %u\n",   m_n_ptw);
  fprintf(f, "  PTW occupancy    : %.2f / %u (avg per cycle)\n",
          (double)m_ptw_active_sum / total_cycles, m_n_ptw);
  fprintf(f, "  PTW saturation   : %.1f%%\n",
          100.0 * (double)m_ptw_full_cycles / total_cycles);
  fprintf(f, "  PTW full cycles  : %llu\n", m_ptw_full_cycles);
  // Raw occupancy integrand/denominator so the per-walk event log
  // (ACCELSIM_LOG_GMMU) can recompute occupancy = sum(walk durations)/cycles.
  fprintf(f, "  cycles counted   : %llu\n", m_total_cycles);
  fprintf(f, "  PTW active sum   : %llu\n", m_ptw_active_sum);

  unsigned long long n_done = m_ptw_complete_count > 0 ? m_ptw_complete_count : 1;
  double avg_queuing = (double)m_pwq_queuing_delay_sum / n_deq;
  double avg_ptw     = (double)m_ptw_time_sum          / n_done;

  fprintf(f, "  avg queuing delay  : %.2f cycles\n", avg_queuing);
  fprintf(f, "  avg PTW time       : %.2f cycles\n", avg_ptw);
  fprintf(f, "  avg total latency  : %.2f cycles\n", avg_queuing + avg_ptw);
  fprintf(f, "  completed walks    : %llu\n",         m_ptw_complete_count);

  fprintf(f, "  PWQ max depth    : %llu\n", m_pwq_max_depth_seen);
  // XLATLAT: PWQ inter-arrival shape. Exponential (i.e. the model's Poisson
  // assumption) would show a geometric decay across bins; a spike at b0/b1 means
  // arrivals come in bursts and M/G/32 does not describe this queue.
  fprintf(f, "  [XLATHIST] chiplet=%u kind=pwq_interarrival n=%llu",
          m_chiplet_id, m_translate_requests);
  xlat_hist_print(f, m_xlat_hist_ia);
  fprintf(f, "\n");

  #ifdef ADDR_XLAT_VERIFY
    // PWC-A: with no walk in flight, every completed walk accounts for exactly
    // 4 levels = DRAM steps (local+remote) + PWC hits.
    if (m_active_walks.empty()) {
      assert(m_local_steps + m_remote_steps ==
            4ULL * m_ptw_complete_count - m_pwc_hits &&
            "V-PTW: PWC-A conservation (local+remote == 4*walks - pwc_hits)");
      fprintf(f, "  [XLAT_VERIFY V-PTW] PWC-A conservation OK : PASS\n");
    }
  #endif
  
  fprintf(f, "  local steps      : %llu\n", m_local_steps);
  fprintf(f, "  remote steps     : %llu\n", m_remote_steps);
  fprintf(f, "  PWC hits         : %llu\n", m_pwc_hits);
  // MEMLAT_ (temp): target sub-partition distribution of PT-read DRAM steps.
  // If remote steps concentrate on 1-2 sub-partitions -> single-partition hotspot.
  fprintf(f, "  [MEMLAT] remote-step sub-partition dist:");
  for (auto &kv : m_memlat_remote_subpart_hist)
    fprintf(f, " sp%u=%llu", kv.first, kv.second);
  fprintf(f, "\n  [MEMLAT] all-step sub-partition dist:");
  for (auto &kv : m_memlat_subpart_hist)
    fprintf(f, " sp%u=%llu", kv.first, kv.second);
  {
    unsigned long long n = m_memlat_seg_cnt ? m_memlat_seg_cnt : 1;
    fprintf(f, "\n  [MEMLAT] per-DRAM-step: steps=%llu  avg_issue->memout=%.1f  "
               "avg_memout->gmmu=%.1f  (sum=%.1f cyc/step)\n",
            m_memlat_seg_cnt,
            (double)m_memlat_seg_reqmem_sum / n,
            (double)m_memlat_seg_reply_sum / n,
            (double)(m_memlat_seg_reqmem_sum + m_memlat_seg_reply_sum) / n);
  }
  // FUNNEL_ (temp): single repr_cluster endpoint funnel evidence (req + reply).
  {
    unsigned long long req_tot = m_funnel_req_ok + m_funnel_req_full;
    fprintf(f, "  [FUNNEL] req: injected=%llu requeued(input_full)=%llu stall=%.1f%%"
               "  | reply: avg=%.1f max=%llu cyc\n",
            m_funnel_req_ok, m_funnel_req_full,
            req_tot ? 100.0 * (double)m_funnel_req_full / req_tot : 0.0,
            m_memlat_seg_cnt ? (double)m_memlat_seg_reply_sum / m_memlat_seg_cnt : 0.0,
            m_memlat_seg_reply_max);
    // FUNNEL_ (temp): reply seg B split — transit(xbar+ring) vs last-hop(dest cluster
    // egress FIFO wait behind normal SM traffic). last-hop dominant => borrowing an
    // SM cluster's icnt egress port is the artifact, not xbar transit.
    unsigned long long sc = m_memlat_seg_split_cnt ? m_memlat_seg_split_cnt : 1;
    fprintf(f, "  [FUNNEL] reply-split: transit_avg=%.1f  lasthop_avg=%.1f (max=%llu)"
               "  [n=%llu]\n",
            (double)m_memlat_seg_transit_sum / sc,
            (double)m_memlat_seg_lasthop_sum / sc,
            m_memlat_seg_lasthop_max, m_memlat_seg_split_cnt);
  }
}
// XLATDBG: what this chiplet's GMMU is still waiting on. A walk that has been
// active for far longer than a DRAM round trip is a lost reply, not a slow one.
void gmmu::xlatdbg_dump_pending(FILE* fp) const {
  const unsigned long long now = (unsigned long long)m_last_cycle;  // gmmu's own clock
  fprintf(fp, "XLATDBG_DEADLOCK: gmmu chiplet=%u active_walks=%zu pwq=%zu pending_steps=%zu\n",
          m_chiplet_id, m_active_walks.size(), m_pwq.size(), m_pending_next_steps.size());
  for (const auto& kv : m_active_walks) {
    mem_fetch* mf = kv.first;
    unsigned long long started = 0;
    auto it = m_walk_start_cycles.find(mf);
    if (it != m_walk_start_cycles.end()) started = it->second.t_walk_start;
    fprintf(fp, "XLATDBG_DEADLOCK:   walk mf=%p level=%d started=%llu age=%llu\n",
            (void*)mf, kv.second, started, (now > started) ? now - started : 0);
  }
}
