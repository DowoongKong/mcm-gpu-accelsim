#include <cstdlib>   // getenv, atoi
#include <cstdint>  // uint64_t, strtoull

#include "gpu-module.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "icnt_wrapper.h"
#include "gmmu.h"  // full definition needed for creating gmmu objects (P3)
#include "gpu_tlb.h"   // gpu_l2_tlb full definition needed for l2_tlb_request()

#include <algorithm>

// Verification logging: per-walk event lines, OFF unless ACCELSIM_LOG_PTW is set.
// Runtime-gated so one binary serves production (no output, digest-neutral)
// and verification runs (recompute walks / local-remote steps / occupancy
// from the log and cross-check the GMMU_STAT counters).
static bool accelsim_log_ptw() {
  static const bool on = (getenv("ACCELSIM_LOG_PTW") != NULL);
  return on;
}


// ===================== Address Translation Verification =====================
// Single toggle. When enabled, runs all active steps once at the end of
// gpu_module::init(). OFF by default -> zero runtime cost in normal builds.
// (plan: docs/verification_plan_address_translation.md)
// #define ADDR_XLAT_VERIFY
#ifdef ADDR_XLAT_VERIFY
  static void run_address_translation_verify(gpu_module* mod);
#endif

const memory_config* gpu_module::get_memory_config() const {
  return m_top->getMemoryConfig();
}

// ------------------------- chiplet -------------------------

chiplet::chiplet(gpgpu_sim* top, unsigned chiplet_id)
    : m_top(top),
      m_id(chiplet_id),
      m_cluster_global_begin(0),
      m_n_clusters_per_chiplet(0),
      m_mem_global_begin(0),
      m_n_mem_per_chiplet(0),
      m_global_clusters(NULL),
      m_global_mem(NULL) {}

chiplet::~chiplet() {
  // Phase 1: chiplet does not own global arrays or the objects inside them.
  // Nothing to delete here.
}

void chiplet::bind_cluster_range(unsigned global_begin, unsigned count) {
  m_cluster_global_begin = global_begin;
  m_n_clusters_per_chiplet = count;
}

void chiplet::bind_mem_range(unsigned global_begin, unsigned count) {
  m_mem_global_begin = global_begin;
  m_n_mem_per_chiplet = count;
  m_mem_sub_partition_global_begin = global_begin * 2; // Assume 2 sub-parts per mem part
  m_n_mem_sub_partition_per_chiplet = count * 2; // Assume 2 sub-parts per mem part
}

void chiplet::bind_global_arrays(simt_core_cluster** global_clusters,
                                 memory_partition_unit** global_mem) {
  m_global_clusters = global_clusters;
  m_global_mem = global_mem;
}

simt_core_cluster* chiplet::get_cluster_global(unsigned global_idx) const {
  assert(m_global_clusters != NULL);
  return m_global_clusters[global_idx];
}

memory_partition_unit* chiplet::get_mem_global(
    unsigned global_idx) const {
  assert(m_global_mem != NULL);
  return m_global_mem[global_idx];
}

simt_core_cluster* chiplet::get_cluster(unsigned local_idx) const {
  assert(m_global_clusters != NULL);
  assert(local_idx < m_n_clusters_per_chiplet);
  return m_global_clusters[m_cluster_global_begin + local_idx];
}

memory_partition_unit* chiplet::get_mem(unsigned local_idx) const {
  assert(m_global_mem != NULL);
  assert(local_idx < m_n_mem_per_chiplet);
  return m_global_mem[m_mem_global_begin + local_idx];
}

// ------------------------- gpu_module -------------------------
gpu_module::gpu_module(gpgpu_sim* top, unsigned n_chiplets, unsigned n_clusters,
                       unsigned n_mem, unsigned n_mem_sub_partition)
    : m_top(top),
      m_n_chiplets(n_chiplets),
      m_n_clusters(n_clusters),
      m_n_mem(n_mem),
      m_n_mem_sub_partition(n_mem_sub_partition),
      m_global_clusters(NULL),
      m_global_mem(NULL),
      m_stats_file(NULL),
      m_remote_access_count(0),
      m_local_access_count(0),
      m_page_table(NULL),
      m_current_lasp_config(NULL),
      m_current_kernel_block_dim(dim3(256, 1, 1)) {
  assert(m_n_chiplets >= 1);
}

gpu_module::~gpu_module() {
  for (size_t i = 0; i < m_chiplets.size(); ++i) {
    delete m_chiplets[i];
  }
  m_chiplets.clear();
  delete m_page_table;  // release page table on simulator teardown

  // P3: release GMMUs
  for (auto* g : m_gmmus) {
    delete g;
  }
  m_gmmus.clear();
}

void gpu_module::init() {
  create_chiplets();

  // Build owner maps.
  m_cluster_to_chiplet.assign(m_n_clusters, 0);
  m_mem_to_chiplet.assign(m_n_mem, 0);
  m_mem_sub_partition_to_chiplet.assign(m_n_mem_sub_partition, 0);

  // Partition clusters and memory partitions with the same policy (contiguous split).
  for (unsigned cid = 0; cid < m_n_chiplets; ++cid) {
    unsigned cb = 0, cc = 0;
    unsigned mb = 0, mc = 0;

    split_range(m_n_clusters, m_n_chiplets, cid, &cb, &cc);
    split_range(m_n_mem, m_n_chiplets, cid, &mb, &mc);

    m_chiplets[cid]->bind_cluster_range(cb, cc);
    m_chiplets[cid]->bind_mem_range(mb, mc);

    for (unsigned i = 0; i < cc; ++i) {
      const unsigned g = cb + i;
      assert(g < m_n_clusters);
      m_cluster_to_chiplet[g] = cid;
    }
    for (unsigned i = 0; i < mc; ++i) {
      const unsigned g = mb + i;
      assert(g < m_n_mem);
      m_mem_to_chiplet[g] = cid;
      // CONSTRAINT: hardcodes 2 sub-partitions per memory channel instead of
      // reading m_n_sub_partition_per_memory_channel (same assumption in
      // chiplet::bind_mem_range). Wrong owner maps if a config ever changes
      // -gpgpu_n_sub_partition_per_mchannel. findings.ko.md SMELL-2.
      m_mem_sub_partition_to_chiplet[2*g] = cid;
      m_mem_sub_partition_to_chiplet[2*g+1] = cid;
    }
  }

  // PMM initialization: split the physical frame space [0, total_pages) into
  // one contiguous ascending range per chiplet. PFNs start at 0 -- the PA space
  // is separate from the VA heap (GLOBAL_HEAP_START); the page table maps
  // between them. Remainder frames go to the LAST chiplet (unlike split_range,
  // which favors lower indices -- the two splits need not agree, they cover
  // different resources).
  uint64_t total_pages = (uint64_t)m_top->get_config().get_gpu_mem_size_mb()
                        * 1024ULL * 1024ULL / GPU_PAGE_SIZE;
  uint64_t pages_per_chiplet         = total_pages / m_n_chiplets;

  for (unsigned cid = 0; cid < m_n_chiplets; cid++) {
    uint64_t pfn_start = (uint64_t)cid * pages_per_chiplet;
    uint64_t pfn_end   = (cid == m_n_chiplets - 1)
                        ? total_pages : pfn_start + pages_per_chiplet;
    // Bump range instead of pushing every frame: same ascending allocation
    // order as the old queue, without materializing millions of entries.
    m_chiplets[cid]->pmm_set_range(pfn_start, pfn_end);
    // A huge page must not straddle chiplets, or pfn_to_chiplet_via_addrdec()
    // would disagree with the page's owner.
    assert(pfn_start % GPU_HUGE_FRAMES == 0 &&
           "PMM: chiplet frame base must be 2MB-aligned for huge pages");
  }

  uint64_t distributed = 0;
  for (unsigned c = 0; c < m_n_chiplets; c++)
    distributed += m_chiplets[c]->free_page_count();
  assert(distributed == total_pages && "PMM page distribution mismatch");

  for (unsigned c = 0; c < m_n_chiplets; c++)
    printf("[PMM] init: chiplet=%u pages=%zu (%.1f%%)\n",
          c, m_chiplets[c]->free_page_count(),
          100.0 * m_chiplets[c]->free_page_count() / total_pages);
  printf("[PMM] init: total=%llu pages across %u chiplets\n",
        (unsigned long long)total_pages, m_n_chiplets);
  
  // Phase 2: create the GPU-wide page table after PMM is ready
  m_page_table = new gpu_page_table(this);
  printf("[PAGE_TABLE] init: gpu_page_table created\n");

  // Optional: freeze placement from a reference run's snapshot and pre-map the
  // whole page table before the first kernel, so the measured region takes zero
  // faults and placement is a controlled variable across configs.
  // Disabled (empty path) by default -> first-touch behaves exactly as before.
  {
    const char* pl = m_top->get_config().get_gmmu_placement_load();
    bool frozen = false;
    if (pl && pl[0] != '\0' && m_page_table->load_placement(pl)) {
      // premap() reserves every data-page PFN first, and only then allocates the
      // root's per-chiplet copies, so data placement is identical with and
      // without -gmmu_local_ptw.
      m_page_table->premap_from_placement();
      frozen = true;
    }
    // Not frozen: nothing has reserved data pages yet, so the root may take its
    // copies now. No-op unless -gmmu_local_ptw.
    if (!frozen) m_page_table->alloc_root_replicas();
  }

  // P3: create one GMMU per chiplet
  unsigned n_ptw       = m_top->get_config().get_gmmu_n_ptw();
  unsigned pwq_depth   = m_top->get_config().get_gmmu_pwq_depth();
  unsigned pwc_entries = m_top->get_config().get_gmmu_pwc_entries();

  for (unsigned c = 0; c < m_n_chiplets; c++) {
    gpu_module* self = m_top->get_gpu_module();
    m_gmmus.push_back(new gmmu(c, n_ptw, pwq_depth, pwc_entries,
                               m_page_table, &m_cluster_to_chiplet, &m_va_alloc_list,
                               self));
    printf("[GMMU] init: chiplet=%u n_ptw=%u pwq_depth=%u pwc_entries=%u\n",
           c, n_ptw, pwq_depth, pwc_entries);
  }
  
  // Create one L2 TLB per chiplet.
  // Give each GMMU a back-pointer so complete_walk() can call l2_tlb->fill().
  for (unsigned c = 0; c < m_n_chiplets; c++) {
    // 4KB L2 TLB. page_shift is now passed explicitly; it used to rely on the
    // GPU_PAGE_SHIFT default.
    gpu_l2_tlb* l2tlb_4k = new gpu_l2_tlb(
        m_top->get_config().get_l2_tlb_sets(),
        m_top->get_config().get_l2_tlb_assoc(),
        m_top->get_config().get_l2_tlb_mshr_entries(),
        m_top->get_config().get_l2_tlb_mshr_max_merge(),
        m_top->get_config().get_l2_tlb_lookup_latency(),
        c,
        GPU_PAGE_SHIFT);
    l2tlb_4k->set_ports(m_top->get_config().get_l2_tlb_ports());
    m_chiplets[c]->set_l2_tlb_4k(l2tlb_4k);
    m_gmmus[c]->set_l2_tlb_4k(l2tlb_4k);
    printf("[L2_TLB_4k] init: chiplet=%u sets=%u assoc=%u mshr_entries=%u\n",
           c,
           m_top->get_config().get_l2_tlb_sets(),
           m_top->get_config().get_l2_tlb_assoc(),
           m_top->get_config().get_l2_tlb_mshr_entries());

    // 64KB L2 TLB: same class, only page_shift differs.
    gpu_l2_tlb* l2tlb_64k = new gpu_l2_tlb(
        m_top->get_config().get_l2_tlb_64k_sets(),
        m_top->get_config().get_l2_tlb_64k_assoc(),
        m_top->get_config().get_l2_tlb_64k_mshr_entries(),
        m_top->get_config().get_l2_tlb_64k_mshr_max_merge(),
        m_top->get_config().get_l2_tlb_64k_lookup_latency(),
        c,
        GPU_64K_SHIFT);
    l2tlb_64k->set_ports(m_top->get_config().get_l2_tlb_ports());
    m_chiplets[c]->set_l2_tlb_64k(l2tlb_64k);
    m_gmmus[c]->set_l2_tlb_64k(l2tlb_64k);
    printf("[L2_TLB_64k] init: chiplet=%u sets=%u assoc=%u mshr_entries=%u\n",
           c,
           m_top->get_config().get_l2_tlb_64k_sets(),
           m_top->get_config().get_l2_tlb_64k_assoc(),
           m_top->get_config().get_l2_tlb_64k_mshr_entries());

    // 2MB L2 TLB: same class, only page_shift differs.
    gpu_l2_tlb* l2tlb_2m = new gpu_l2_tlb(
        m_top->get_config().get_l2_tlb_2m_sets(),
        m_top->get_config().get_l2_tlb_2m_assoc(),
        m_top->get_config().get_l2_tlb_2m_mshr_entries(),
        m_top->get_config().get_l2_tlb_2m_mshr_max_merge(),
        m_top->get_config().get_l2_tlb_2m_lookup_latency(),
        c,
        GPU_HUGE_SHIFT);
    l2tlb_2m->set_ports(m_top->get_config().get_l2_tlb_ports());
    m_chiplets[c]->set_l2_tlb_2m(l2tlb_2m);
    m_gmmus[c]->set_l2_tlb_2m(l2tlb_2m);
    printf("[L2_TLB_2m] init: chiplet=%u sets=%u assoc=%u mshr_entries=%u\n",
           c,
           m_top->get_config().get_l2_tlb_2m_sets(),
           m_top->get_config().get_l2_tlb_2m_assoc(),
           m_top->get_config().get_l2_tlb_2m_mshr_entries());
  }
  
  // inter-chiplet one-way latency: ns -> core cycles (round). 0 keeps legacy behavior.
  {
    unsigned long long ns  = m_top->get_config().get_icnt_inter_hop_latency_ns();
    unsigned long long fhz = m_top->get_config().get_core_freq();   // Hz (e.g. 1.132e9)
    m_inter_hop_cycles = (unsigned)((ns * fhz + 500000000ULL) / 1000000000ULL);
    // P3: ring and the legacy fixed-latency delay queue must not both add inter
    // latency. When the ring is on, set -icnt_inter_hop_latency_ns 0.
    assert(!(icnt_ring_enabled() && m_n_chiplets > 1 && m_inter_hop_cycles > 0) &&
          "ring + m_inter_delay_q both active: inter latency double-counted; "
          "set -icnt_inter_hop_latency_ns 0 when -icnt_use_ring 1");
    printf("[ICNT] inter-chiplet one-way latency = %llu ns = %u core cycles\n",
           ns, m_inter_hop_cycles);
  }

  #ifdef ADDR_XLAT_VERIFY
  run_address_translation_verify(m_top->get_gpu_module());  // no-this rule: self via top
  #endif
}


void gpu_module::create_chiplets() {
  if (!m_chiplets.empty()) return;

  m_chiplets.resize(m_n_chiplets, NULL);
  for (unsigned i = 0; i < m_n_chiplets; ++i) {
    m_chiplets[i] = new chiplet(m_top, i);
  }
}

void gpu_module::split_range(unsigned total, unsigned n_mem, unsigned mem_id,
                             unsigned* out_begin, unsigned* out_count) {
  assert(n_mem >= 1);
  assert(mem_id < n_mem);
  assert(out_begin != NULL);
  assert(out_count != NULL);

  const unsigned base = total / n_mem;
  const unsigned rem  = total % n_mem;

  // Remainder is distributed to lower-index partition.
  const unsigned extra = (mem_id < rem) ? 1 : 0;

  // Begin = part_id*base + min(part_id, rem)
  const unsigned begin = mem_id * base + ((mem_id < rem) ? mem_id : rem);
  const unsigned count = base + extra;

  *out_begin = begin;
  *out_count = count;
}

void gpu_module::bind_global_arrays(simt_core_cluster** global_clusters,
                                   memory_partition_unit** global_mem) {
  assert(global_clusters != NULL);
  assert(global_mem != NULL);

  m_global_clusters = global_clusters;
  m_global_mem = global_mem;

  // Bind global pointers into each chiplet.
  for (unsigned cid = 0; cid < m_n_chiplets; ++cid) {
    m_chiplets[cid]->bind_global_arrays(m_global_clusters, m_global_mem);
  }
}

void gpu_module::icnt_deliver(unsigned input_deviceID,
                              unsigned output_deviceID,
                              void* data,
                              unsigned int size) {
  mem_fetch* mf = static_cast<mem_fetch*>(data);

  // PTW reply travels back via the reply-ring so the return trip pays inter-chiplet
  // latency (do NOT short-circuit at source). It is intercepted at the destination
  // pop (gpu_module::icnt_pop) and routed to that chiplet's GMMU.
  if (mf->get_type() == TLB_FILL_REPLY) {
    unsigned src_chiplet = mem_sub_partition_to_chiplet(device2mem(input_deviceID));
    unsigned dst_chiplet = cluster_to_chiplet(device2cluster(output_deviceID));
    mf->set_local_access(src_chiplet == dst_chiplet);
    FILE* log_file = m_stats_file ? m_stats_file : stderr;
    if (accelsim_log_ptw()) {
    fprintf(log_file, "[MCM_PTW] chiplet%u -> chiplet%u : %s (reply)\n",
            src_chiplet, dst_chiplet,
            (src_chiplet != dst_chiplet) ? "REMOTE" : "LOCAL");
    }
    ::icnt_push(input_deviceID, output_deviceID, data, size);
    return;
  }

  // PTW outgoing request: log as PTW access (not regular SM access)
  if (mf->get_type() == TLB_FILL_ACC) {
    unsigned dst_chiplet = mem_sub_partition_to_chiplet(device2mem(output_deviceID));
    bool remote = (cluster_to_chiplet(device2cluster(input_deviceID)) != dst_chiplet);
    mf->set_local_access(!remote);
    FILE* log_file = m_stats_file ? m_stats_file : stderr;
    if (accelsim_log_ptw()) {
    fprintf(log_file, "[MCM_PTW] chiplet%u -> chiplet%u : %s\n",
            cluster_to_chiplet(device2cluster(input_deviceID)),
            dst_chiplet, remote ? "REMOTE" : "LOCAL");
    }
    ::icnt_push(input_deviceID, output_deviceID, data, size);
    return;
  }
  
  bool remote = is_remote_access(input_deviceID, output_deviceID, data);

  if (remote) {
    mf->set_local_access(false);
    // 44-2 (V6): bumped from the parallel CORE phase B (each chiplet's SMs push
    // through here concurrently); non-atomic increments were losing counts
    // (mcm_local_accesses differed run-to-run). Statistic only, model-neutral.
#pragma omp atomic
    m_remote_access_count++;
  } else {
    mf->set_local_access(true);
#pragma omp atomic
    m_local_access_count++;
  }

  // Log to stats file instead of stderr
  FILE* log_file = m_stats_file ? m_stats_file : stderr;

  if (mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_REQUEST) {
    unsigned clusterid = device2cluster(input_deviceID);
    unsigned memid     = device2mem(output_deviceID);
    unsigned src_chiplet_id = cluster_to_chiplet(clusterid);
    unsigned dst_chiplet_id = mem_sub_partition_to_chiplet(memid);

    #ifdef LOG_MEM_ACCESS
    fprintf(log_file,
            "[MCM_ACCESS] chiplet%u cluster%u -> chiplet%u mem%u : %s\n",
            src_chiplet_id, clusterid, dst_chiplet_id, memid,
            remote ? "REMOTE" : "LOCAL");
    #endif
  } else if (mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK) {
    unsigned memid     = device2mem(input_deviceID);
    unsigned clusterid = device2cluster(output_deviceID);
    unsigned src_chiplet_id = mem_sub_partition_to_chiplet(memid);
    unsigned dst_chiplet_id = cluster_to_chiplet(clusterid);

    #ifdef LOG_MEM_ACCESS
    fprintf(log_file,
            "[MCM_ACCESS] chiplet%u mem%u -> chiplet%u cluster%u : %s\n",
            src_chiplet_id, memid, dst_chiplet_id, clusterid,
            remote ? "REMOTE" : "LOCAL");
    #endif
  }

  ::icnt_push(input_deviceID, output_deviceID, data, size);
}

void gpu_module::icnt_push(unsigned input_deviceID, unsigned output_deviceID,
                           void* data, unsigned int size) {
  mem_fetch* mf = static_cast<mem_fetch*>(data);
  if (m_inter_hop_cycles > 0 && is_inter_chiplet_xing(input_deviceID, output_deviceID, mf)) {
    uint64_t now = m_top->gpu_sim_cycle + m_top->gpu_tot_sim_cycle;
    m_inter_delay_q.push({input_deviceID, output_deviceID, size, data,
                          now + m_inter_hop_cycles});
    return;                          // held on the inter-chiplet link; delivered later
  }
  icnt_deliver(input_deviceID, output_deviceID, data, size);   // local or latency off
}

bool gpu_module::is_inter_chiplet_xing(unsigned in, unsigned out, mem_fetch* mf) const {
  switch (mf->get_type()) {
    case READ_REQUEST: case WRITE_REQUEST: case TLB_FILL_ACC:   // cluster -> mem
      return cluster_to_chiplet(device2cluster(in)) !=
             mem_sub_partition_to_chiplet(device2mem(out));
    case READ_REPLY: case WRITE_ACK: case TLB_FILL_REPLY:       // mem -> cluster
      return mem_sub_partition_to_chiplet(device2mem(in)) !=
             cluster_to_chiplet(device2cluster(out));
    default:
      return false;
  }
}

void gpu_module::advance_inter_chiplet(uint64_t now) {
  while (!m_inter_delay_q.empty() && m_inter_delay_q.front().release <= now) {
    delayed_pkt_t p = m_inter_delay_q.front();
    mem_fetch* mf = static_cast<mem_fetch*>(p.data);
    // PTW reply is intercepted (done_tlb_req) and never enters the network buffer.
    // For everything else, respect the network input buffer (head-of-line block).
    if (mf->get_type() != TLB_FILL_REPLY && !icnt_has_buffer(p.in, p.size))
      break;
    m_inter_delay_q.pop();
    icnt_deliver(p.in, p.out, p.data, p.size);
  }
}

// P5: L2 TLB request from shader core. Returns true if request is accepted (either hit or MSHR allocated).
// link_used (out): did this call actually put a request on the SM->L2 link?
// A request already inside the L2's lookup stage is NOT retransmitted -- it is
// simply still being probed -- so it must not consume issue bandwidth. Charging
// it would recreate the head-of-line block this drain exists to remove.
bool gpu_module::l2_tlb_request(uint64_t va, unsigned cluster_id,
                                  uint64_t cycle,
                                  gpu_pte_t& out_pte, bool& is_l2_hit,
                                  bool* link_used, uint64_t* pending_until,
                                  unsigned* full_kind) {
  unsigned    chiplet_id = cluster_to_chiplet(cluster_id);
  // Route to the L2 TLB that matches this VA's page size.
  const unsigned pshift  = page_shift_for_va(va);
  gpu_l2_tlb* l2tlb      = m_chiplets[chiplet_id]->get_l2_tlb(pshift);
  gmmu*       gmmu_ptr   = m_gmmus[chiplet_id];

  is_l2_hit = false;
  // Default: this call transmitted something (a new request or a rejected one).
  if (link_used) *link_used = true;

  gpu_pte_t l2_pte;
  const bool was_probing = l2tlb->is_probing(cluster_id, page_vpn(va, pshift));
  auto result = l2tlb->access(va, cluster_id, l2_pte, cycle, pending_until);
  // Already inside the lookup stage before this call -> nothing was sent.
  if (was_probing && link_used) *link_used = false;

  switch (result) {
    case gpu_l2_tlb::HIT:
      // L2 TLB array hit: return pte immediately.
      // Caller (shader.cc) fills L1 TLB directly without polling.
      out_pte   = l2_pte;
      is_l2_hit = true;
      return true;

    case gpu_l2_tlb::MSHR_FULL:
      // L2 MSHR at capacity for a new VPN; no MSHR entry was created.
      // SM stalls and retries next cycle.
      if (full_kind) *full_kind = l2tlb->last_full_merge() ? 1u : 2u;  // 44-4
      #ifdef LOG_TLB
      printf("[L2_TLB] chiplet=%u cluster=%u vpn=0x%llx -> MSHR_FULL\n",
               chiplet_id, cluster_id, (unsigned long long)page_vpn(va, pshift));
      #endif
      return false;

    case gpu_l2_tlb::MISS:
      // First miss for this VPN on this chiplet: start PTW via GMMU.
      // access() already created the L2 MSHR entry.
      if (gmmu_ptr->pwq_full()) {
        // Cannot start walk; roll back L2 MSHR add so SM retries cleanly.
        // XLATDBG: the gate was already erased by access(); rolling back only the
        // MSHR leaves the request to re-arm a fresh full-latency gate next retry.
        m_xlatdbg_pwq_rollback++;
        l2tlb->remove_from_mshr(page_vpn(va, pshift), cluster_id);
        return false;
      }
      // Use chiplet representative cluster_id so demand-paging in
      // done_tlb_req() allocates PT nodes and data pages on this chiplet.
      gmmu_ptr->translate(va,
          m_chiplets[chiplet_id]->get_cluster_global_begin(), cycle);
      // XLATLAT: t5 -- the PWQ accepted it. Keyed by the REQUESTING cluster, not
      // the chiplet representative that translate() takes for page placement.
      l2tlb->xlat_mark_pwq_enq(cluster_id, page_vpn(va, pshift), cycle);
      #ifdef LOG_TLB
      printf("[L2_TLB] chiplet=%u cluster=%u vpn=0x%llx -> MISS\n",
               chiplet_id, cluster_id, (unsigned long long)page_vpn(va, pshift));
      #endif
      return true;

    case gpu_l2_tlb::MSHR_HIT:
      // VPN already in-flight; cluster_id merged into existing MSHR entry.
      #ifdef LOG_TLB
      printf("[L2_TLB] chiplet=%u cluster=%u vpn=0x%llx -> MSHR_HIT\n",
               chiplet_id, cluster_id, (unsigned long long)page_vpn(va, pshift));
      #endif
      return true;
    
    case gpu_l2_tlb::LOOKUP_PENDING:
      return false;   // lookup latency still elapsing -> SM stalls and retries

    case gpu_l2_tlb::PORT_BUSY:
      // All L2 lookup ports occupied. Nothing was armed and no resource is held,
      // so the caller just retries; this is a bandwidth limit, distinct from
      // LOOKUP_PENDING (which means THIS request is already being probed).
      return false;

    default:
      assert(0 && "unknown l2_tlb result");
      return false;
  }
}


// void* gpu_module::icnt_pop(unsigned output_deviceID) {
//   return ::icnt_pop(output_deviceID);
// }

void* gpu_module::icnt_pop(unsigned output_deviceID) {
  void* data = ::icnt_pop(output_deviceID);
  if (data) {
    mem_fetch* mf = static_cast<mem_fetch*>(data);
    // PTW reply arrived at the requester chiplet via the reply-ring (round-trip
    // latency paid). Route to this chiplet's GMMU; hide from the SM.
    if (mf->get_type() == TLB_FILL_REPLY) {
      unsigned chiplet_id = cluster_to_chiplet(device2cluster(output_deviceID));
      uint64_t cur = m_top->gpu_sim_cycle + m_top->gpu_tot_sim_cycle;
      m_gmmus[chiplet_id]->done_tlb_req(mf, cur);   // 실제 도착 cycle = ring 왕복 반영
      return nullptr;                               // SM에는 숨김
    }
  }
  return data;
}

bool gpu_module::is_remote_access(unsigned inputDeviceID,
                                  unsigned outputDeviceID,
                                  void* data) const {
                              
  mem_fetch* mf = static_cast<mem_fetch*>(data);

  if (mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_REQUEST) {
    // Shader -> Mem partition
    unsigned clusterid = device2cluster(inputDeviceID);
    unsigned memid = device2mem(outputDeviceID);
    unsigned src_chiplet_id = cluster_to_chiplet(clusterid);
    unsigned dst_chiplet_id = mem_sub_partition_to_chiplet(memid);

    return (src_chiplet_id != dst_chiplet_id);
  } else if (mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK) {
    // Mem partition -> Shader
    unsigned memid = device2mem(inputDeviceID);
    unsigned clusterid = device2cluster(outputDeviceID);
    unsigned src_chiplet_id = mem_sub_partition_to_chiplet(memid);
    unsigned dst_chiplet_id = cluster_to_chiplet(clusterid);

    return (src_chiplet_id != dst_chiplet_id);
  }
  // TLB_FILL and other control messages route through icnt;
  // chiplet routing is handled by the address itself, not by src/dst IDs.
  return false;  //new
}


unsigned gpu_module::cluster_to_chiplet(unsigned global_id) const {
  assert(global_id < m_cluster_to_chiplet.size());
  return m_cluster_to_chiplet[global_id];
}

unsigned gpu_module::mem_sub_partition_to_chiplet(unsigned global_id) const {
  // // Always check LASP config first, regardless of policy
  // if (m_current_lasp_config) {
  //   lasp::scheduling_policy_t policy = m_current_lasp_config->policy;
    
  //   if (policy == lasp::ROW_BINDING) {
  //     // Row-based memory allocation
  //     unsigned row = global_id / 4;
  //     unsigned chiplet = (row / 2) % m_n_chiplets;
  //     // fprintf(stdout, "[LASP-MEM-ROW] global_id=%u -> chiplet=%u\n", global_id, chiplet);
  //     fflush(stdout);
  //     return chiplet;
  //   } 
  //   else if (policy == lasp::COL_BINDING) {
  //     // Column-based memory allocation
  //     unsigned col = global_id % 4;
  //     unsigned chiplet = (col / 2) % m_n_chiplets;
  //     // fprintf(stdout, "[LASP-MEM-COL] global_id=%u -> chiplet=%u\n", global_id, chiplet);
  //     fflush(stdout);
  //     return chiplet;
  //   } 
  //   else if (policy == lasp::CHUNK_BINDING) {
  //     // Chunk-based memory allocation: distribute by CTA (not by thread ID)
  //     // Calculate threads per CTA from current kernel's block dimension
  //     unsigned threads_per_cta = m_current_kernel_block_dim.x * 
  //                                 m_current_kernel_block_dim.y * 
  //                                 m_current_kernel_block_dim.z;
      
  //     if (threads_per_cta == 0) threads_per_cta = 256;  // Fallback to default
      
  //     unsigned cta_id = global_id / threads_per_cta;
  //     unsigned chiplet = (cta_id / m_current_lasp_config->chunk_size) % m_n_chiplets;
  //     // fprintf(stdout, "[LASP-MEM-CHUNK] global_id=%u (CTA=%u, threads/cta=%u) -> chiplet=%u\n", 
  //     //        global_id, cta_id, threads_per_cta, chiplet);
  //     fflush(stdout);
  //     return chiplet;
  //   }
  //   else if (policy == lasp::KERNEL_WIDE) {
  //     //fprintf(stdout, "[LASP-MEM-DEFAULT] global_id=%u using default\n", global_id);
  //     fflush(stdout);
  //   }
  // }

  // Look up physical chiplet ownership from the static mapping table.
  // This table is set in init() based on the address mapping and is the
  // authoritative source for which chiplet owns a given sub-partition.
  // LASP is a CTA scheduling policy and does not change physical DRAM ownership.
  // Default behavior: use pre-computed mapping
  assert(global_id < m_mem_sub_partition_to_chiplet.size());
  return m_mem_sub_partition_to_chiplet[global_id];
}

void gpu_module::dump_partitioning(FILE* fout) const {
  if (fout == NULL) fout = stdout;

  fprintf(fout, "======== gpu_module partitioning ========\n");
  fprintf(fout, "num_chiplets=%u, num_clusters=%u, num_mem=%u\n",
          m_n_chiplets, m_n_clusters, m_n_mem);

  for (unsigned cid = 0; cid < m_n_chiplets; ++cid) {
    const chiplet* c = m_chiplets[cid];
    fprintf(fout,
            "chiplet[%u]: clusters=[%u..%u) count=%u, mem=[%u..%u) count=%u, mem_sub_partition=[%u..%u) count=%u\n",
            cid,
            c->get_cluster_global_begin(),
            c->get_cluster_global_begin() + c->get_num_clusters(),
            c->get_num_clusters(),
            c->get_mem_global_begin(),
            c->get_mem_global_begin() + c->get_num_mem(),
            c->get_num_mem(),
            c->get_mem_sub_partition_global_begin(),
            c->get_mem_sub_partition_global_begin() + c->get_num_mem_sub_partition(),
            c->get_num_mem_sub_partition());
  }

  fprintf(fout, "==========================================\n");
}

// void gpu_module::allocate_memory_partitions_lasp() {
//   // Allocate memory partitions to chiplets based on LASP policy
//   if (!m_current_lasp_config) {
//     fprintf(stdout, "[LASP-MEMORY] No LASP config - using default allocation\n");
//     fflush(stdout);
//     return;
//   }

//   fprintf(stdout, "[LASP-MEMORY] Allocating memory partitions based on %s policy\n",
//           lasp::policy_to_string(m_current_lasp_config->policy));
//   fflush(stdout);

//   unsigned n_mem_partitions = m_n_mem;
//   unsigned partitions_per_chiplet = (n_mem_partitions + m_n_chiplets - 1) / m_n_chiplets;

//   if (m_current_lasp_config->policy == lasp::CHUNK_BINDING) {
//     // Allocate memory partitions in round-robin fashion across chiplets
//     for (unsigned mem_id = 0; mem_id < n_mem_partitions; ++mem_id) {
//       unsigned chiplet_id = mem_id % m_n_chiplets;
      
//       // Update sub-partition mapping (2 sub-partitions per mem partition)
//       m_mem_sub_partition_to_chiplet[2 * mem_id] = chiplet_id;
//       m_mem_sub_partition_to_chiplet[2 * mem_id + 1] = chiplet_id;
      
//       fprintf(stdout, "[LASP-MEMORY] mem_partition[%u] -> chiplet %u\n", 
//               mem_id, chiplet_id);
//     }
//   }
//   else if (m_current_lasp_config->policy == lasp::ROW_BINDING) {
//     // Allocate based on row distribution
//     for (unsigned mem_id = 0; mem_id < n_mem_partitions; ++mem_id) {
//       unsigned chiplet_id = (mem_id / 4) % m_n_chiplets;
      
//       m_mem_sub_partition_to_chiplet[2 * mem_id] = chiplet_id;
//       m_mem_sub_partition_to_chiplet[2 * mem_id + 1] = chiplet_id;
      
//       fprintf(stdout, "[LASP-MEMORY] mem_partition[%u] -> chiplet %u (ROW)\n", 
//               mem_id, chiplet_id);
//     }
//   }
//   else if (m_current_lasp_config->policy == lasp::COL_BINDING) {
//     // Allocate based on column distribution
//     for (unsigned mem_id = 0; mem_id < n_mem_partitions; ++mem_id) {
//       unsigned chiplet_id = (mem_id % 4) % m_n_chiplets;
      
//       m_mem_sub_partition_to_chiplet[2 * mem_id] = chiplet_id;
//       m_mem_sub_partition_to_chiplet[2 * mem_id + 1] = chiplet_id;
      
//       fprintf(stdout, "[LASP-MEMORY] mem_partition[%u] -> chiplet %u (COL)\n", 
//               mem_id, chiplet_id);
//     }
//   }

//   fflush(stdout);
// }

void gpu_module::allocate_memory_partitions_lasp() {
  // No-op for baseline MCM-GPU (demand paging + First Touch + Round Robin CTA).
  //
  // m_mem_sub_partition_to_chiplet[] encodes the physical chiplet ownership
  // derived from the address mapping and is set correctly in init().
  // LASP is a CTA scheduling policy and must not alter this physical mapping.
  //
  // LASP-aware memory placement (e.g., co-locating CTA data with its chiplet)
  // belongs in the page placement layer (page_placement_policy), not here.
  return;
}

void gpu_module::dump_mapping_ranges(FILE* fout,
                              const char* tag,
                              const std::vector<unsigned>& chiplet_mapper) const {
  if (!fout) fout = stdout;
  if (chiplet_mapper.empty()) return;

  unsigned cur_chiplet = chiplet_mapper[0];
  unsigned begin = 0;

  for (unsigned i = 1; i < chiplet_mapper.size(); ++i) {
    if (chiplet_mapper[i] != cur_chiplet) {
      fprintf(fout, "%s: [%u..%u] -> chiplet %u\n", tag, begin, i - 1, cur_chiplet);
      begin = i;
      cur_chiplet = chiplet_mapper[i];
    }
  }

  fprintf(fout, "%s: [%u..%u] -> chiplet %u\n",
          tag, begin, (unsigned)chiplet_mapper.size() - 1, cur_chiplet);
}


void gpu_module::dump_mapping(unsigned sub_partition_per_mem,
                                 bool verbose_per_id,
                                 FILE* fout) const {
  if (!fout) fout = stdout;

  fprintf(fout, "==== gpu_module owner maps (Phase 1) ====\n");

  // Cluster -> chiplet
  if (!verbose_per_id) {
    dump_mapping_ranges(fout, "cluster", m_cluster_to_chiplet);
  } else {
    for (unsigned c = 0; c < m_cluster_to_chiplet.size(); ++c) {
      fprintf(fout, "cluster[%u] -> chiplet %u\n", c, m_cluster_to_chiplet[c]);
    }
  }

  // DRAM(mem partition) -> chiplet
  if (!verbose_per_id) {
    dump_mapping_ranges(fout, "mem_part", m_mem_to_chiplet);
  } else {
    for (unsigned p = 0; p < m_mem_to_chiplet.size(); ++p) {
      fprintf(fout, "mem_part[%u] -> chiplet %u\n", p, m_mem_to_chiplet[p]);
    }
  }

  // L2 slice -> chiplet (inherits mem partition owner in Phase 1)
  if (sub_partition_per_mem > 0 && !m_mem_to_chiplet.empty()) {
    const unsigned total_l2 =
        (unsigned)m_mem_to_chiplet.size() * sub_partition_per_mem;

    if (!verbose_per_id) {
      unsigned begin = 0;
      unsigned cur_chiplet = m_mem_to_chiplet[0];
      for (unsigned lid = 1; lid < total_l2; ++lid) {
        const unsigned chiplet = m_mem_to_chiplet[lid / sub_partition_per_mem];
        if (chiplet != cur_chiplet) {
          fprintf(fout, "l2_slice: [%u..%u] -> chiplet %u\n",
                  begin, lid - 1, cur_chiplet);
          begin = lid;
          cur_chiplet = chiplet;
        }
      }

      fprintf(fout, "l2_slice: [%u..%u] -> chiplet %u\n",
              begin, total_l2 - 1, cur_chiplet);
    } else {
      for (unsigned lid = 0; lid < total_l2; ++lid) {
        const unsigned p = lid / sub_partition_per_mem;
        const unsigned sp = lid % sub_partition_per_mem;
        const unsigned chiplet = m_mem_to_chiplet[p];
        fprintf(fout, "l2_slice[%u] (mem_part=%u sub=%u) -> chiplet %u\n",
                lid, p, sp, chiplet);
      }
    }
  }

  fprintf(fout, "========================================\n");
}

// new
uint64_t gpu_module::pmm_alloc_page(unsigned chiplet_id) {
  uint64_t pfn = m_chiplets[chiplet_id]->alloc_page();
  #ifdef LOG_PMM
    printf("[PMM] alloc: chiplet=%u pfn=0x%llx free_remaining=%zu\n",
           chiplet_id, (unsigned long long)pfn,
           m_chiplets[chiplet_id]->free_page_count());
  #endif
  return pfn;
}

uint64_t gpu_module::pmm_alloc_contiguous(unsigned chiplet_id, uint64_t n_frames) {
  uint64_t pfn = m_chiplets[chiplet_id]->alloc_contiguous(n_frames);
  #ifdef LOG_PMM
  printf("[PMM] alloc_contiguous: chiplet=%u n=%llu base_pfn=0x%llx free=%zu\n",
         chiplet_id, (unsigned long long)n_frames, (unsigned long long)pfn,
         m_chiplets[chiplet_id]->free_page_count());
  #endif
  return pfn;
}

void gpu_module::pmm_free_page(uint64_t pfn, unsigned chiplet_id) {
  m_chiplets[chiplet_id]->free_page(pfn);
}

const gpu_pte_t* gpu_module::walk_va(uint64_t va, int* walk_depth) const {
  assert(m_page_table != NULL);
  return m_page_table->walk(va, walk_depth);
}


// MISNOMER (kept for now, findings.ko.md SMELL-3): despite the name this does
// NOT consult addrdec -- it splits the PFN space linearly, mirroring exactly
// the per-chiplet ranges pmm_set_range() built in init() (remainder to the
// last chiplet). It answers "which chiplet's PMM owns this frame", which is
// only the same thing as "which chiplet's DRAM channel serves this PA" if the
// config's addrdec chip bits agree with the linear split -- that agreement is
// a config obligation, not enforced here. An earlier addrdec-based version
// was replaced by this one; see git history.
unsigned gpu_module::pfn_to_chiplet_via_addrdec(uint64_t pa) const {
  uint64_t total_pages       = (uint64_t)m_top->get_config().get_gpu_mem_size_mb()
                               * 1024ULL * 1024ULL / GPU_PAGE_SIZE;
  uint64_t pages_per_chiplet = total_pages / m_n_chiplets;
  uint64_t pfn               = pa >> GPU_PAGE_SHIFT;
  unsigned cid               = (unsigned)(pfn / pages_per_chiplet);
  return (cid < m_n_chiplets) ? cid : m_n_chiplets - 1;
}

// ========================= gpu_page_table =========================

gpu_page_table::gpu_page_table(gpu_module* mod)
    : m_mod(mod), m_pml4(NULL) {
  // Allocate PML4 root immediately, mirroring real hardware where CR3 is set
  // at process creation time before any memory access occurs.
  // with_replicas=false: the root's per-chiplet copies are allocated later by
  // alloc_root_replicas(), AFTER premap reserves the data PFNs. Allocating them
  // here would drain one page from every chiplet pool first and shift every data
  // PFN by one, breaking the "same data placement" invariant.
  // (local_ptw_experiment_design.ko.md 2.4 -- a trap that was actually hit.)
  m_pml4 = alloc_node(0, /*with_replicas=*/false);
  assert(m_pml4 != NULL && "gpu_page_table: failed to allocate PML4 root");
}

gpu_page_table::~gpu_page_table() {
  // Free root
  delete m_pml4;

  // Free all intermediate nodes
  for (auto& kv : m_pdp_nodes)   delete kv.second;
  for (auto& kv : m_pd_nodes)    delete kv.second;
  for (auto& kv : m_pt_nodes)    delete kv.second;
  for (auto& kv : m_pt64k_nodes) delete kv.second;
}

bool gpu_module::local_ptw_enabled() const {
  return m_top->get_config().get_gmmu_local_ptw();
}

unsigned gpu_module::uniform_page_shift() const {
  // Fixed policies (0-2) answer the same shift for any size; automatic policies
  // (3-5) do not. Probing the two extremes distinguishes them without having to
  // duplicate the policy table here.
  const unsigned small = m_top->decide_page_shift(1);
  const unsigned big   = m_top->decide_page_shift(GPU_HUGE_PAGE_SIZE);
  return (small == big) ? small : 0;
}

// Give `node` one page per OTHER chiplet so every chiplet can walk it locally.
// No-op unless -gmmu_local_ptw. The pages are genuinely consumed, so
// pt_page_count() reports the n_chiplets-fold cost.
bool gpu_page_table::alloc_replicas(pt_node_t* node, unsigned owner) {
  if (!m_mod->local_ptw_enabled()) return true;   // baseline: nothing to do
  const unsigned n_chip = m_mod->get_num_chiplets();
  for (unsigned c = 0; c < n_chip; c++) {
    if (c == owner) continue;                              // owner has its copy
    if (node->base_pa_rep[c] != node->base_pa) continue;   // already replicated
    uint64_t rep_pfn = m_mod->pmm_alloc_page(c);
    if (rep_pfn == (uint64_t)-1) {
      printf("[PAGE_TABLE] alloc_node: PMM exhausted on chiplet %u (replica)\n", c);
      return false;
    }
    node->base_pa_rep[c] = rep_pfn << GPU_PAGE_SHIFT;
  }
  return true;
}

void gpu_page_table::alloc_root_replicas() {
  assert(m_pml4 != nullptr);
  alloc_replicas(m_pml4, 0);   // root's owner chiplet is 0
}

// PT node pages actually consumed from the PMM.
size_t gpu_page_table::pt_page_count() const {
  size_t nodes = pt_node_count();
  return m_mod->local_ptw_enabled() ? nodes * m_mod->get_num_chiplets() : nodes;
}

pt_node_t* gpu_page_table::alloc_node(unsigned chiplet_id, bool with_replicas) {
  // Allocate one 4KB page from the target chiplet's PMM for this PT node.
  // A PT node page is 4KB for every page size -- a 64KB leaf node is still a
  // 512-entry table, not a 64KB data page.
  uint64_t pfn = m_mod->pmm_alloc_page(chiplet_id);
  if (pfn == (uint64_t)-1) {
    printf("[PAGE_TABLE] alloc_node: PMM exhausted on chiplet %u\n", chiplet_id);
    return NULL;
  }
  pt_node_t* node = new pt_node_t();
  node->base_pa = pfn << GPU_PAGE_SHIFT;  // record PA for Phase 6 PTW latency

  // Per-chiplet view. Baseline: every chiplet sees the single page just
  // allocated, so a foreign-chiplet walk reads it remotely -- exactly the
  // behaviour before this option existed.
  node->base_pa_rep.assign(m_mod->get_num_chiplets(), node->base_pa);

  // Replicating here (rather than in a later bulk pass) means EVERY node, at
  // every level and for every page size -- PML4/PDP/PD/PT/PT64K -- is covered
  // automatically, because they all come through this one function.
  if (with_replicas && !alloc_replicas(node, chiplet_id)) return NULL;
  return node;
}

const gpu_pte_t* gpu_page_table::walk(uint64_t va, int* walk_depth) const {
  int depth = 0;
  if (walk_depth) *walk_depth = 0;

  // Level 0: PML4
  if (!m_pml4) return NULL;
  const gpu_pte_t* pml4e = &m_pml4->entries[pt_pml4_idx(va)];
  depth++;
  if (!pml4e->present()) {
    if (walk_depth) *walk_depth = depth;
    return NULL;
  }

  // Level 1: PDP
  uint64_t pdp_key = pt_pml4_idx(va);
  auto pdp_it = m_pdp_nodes.find(pdp_key);
  if (pdp_it == m_pdp_nodes.end()) return NULL;
  const gpu_pte_t* pdpe = &pdp_it->second->entries[pt_pdp_idx(va)];
  depth++;
  if (!pdpe->present()) {
    if (walk_depth) *walk_depth = depth;
    return NULL;
  }

  // Level 2: PD
  uint64_t pd_key = (pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
  auto pd_it = m_pd_nodes.find(pd_key);
  if (pd_it == m_pd_nodes.end()) return NULL;
  const gpu_pte_t* pde = &pd_it->second->entries[pt_pd_idx(va)];
  depth++;
  if (!pde->present()) {
    if (walk_depth) *walk_depth = depth;
    return NULL;
  }
  // 2MB huge page: PS=1 makes this PDE the leaf, so there is no PT level.
  // Decided from the PTE bit alone, exactly like a hardware walk would.
  if (pde->huge_page()) {
    if (walk_depth) *walk_depth = depth;  // 3: PML4 -> PDP -> PD
    return pde;
  }

  // Level 3: PT (leaf). 4KB and 64KB both land here but read different nodes:
  // the 512-entry 4KB PT vs the 32-entry 64KB big PT. Unlike the 2MB case there
  // is no PTE bit to tell them apart (real HW follows the PDE's small/big pointer
  // pair), so the page size comes from the VA's allocation record.
  const bool is_64k = (m_mod->page_shift_for_va(va) == GPU_64K_SHIFT);
  uint64_t pt_key = ((uint64_t)pt_pml4_idx(va) << 18) |
                    ((uint64_t)pt_pdp_idx(va)   <<  9) |
                     (uint64_t)pt_pd_idx(va);
  const auto& nodes = is_64k ? m_pt64k_nodes : m_pt_nodes;
  auto pt_it = nodes.find(pt_key);
  if (pt_it == nodes.end()) return NULL;
  const gpu_pte_t* pte = is_64k ? &pt_it->second->entries[pt_pt64k_idx(va)]
                                : &pt_it->second->entries[pt_pt_idx(va)];
  depth++;
  if (walk_depth) *walk_depth = depth;

  return pte->present() ? pte : NULL;
}

// void gpu_page_table::map(uint64_t va, unsigned chiplet_id) {
//   // PML4: single global root for the entire GPU, always on chiplet 0.
//   // Every GMMU must access it at the start of every PTW regardless of
//   // which chiplet owns the data, so one unavoidable remote hop per walk.
//   assert(m_pml4 != NULL);  // always pre-allocated in constructor

//   gpu_pte_t* pml4e = &m_pml4->entries[pt_pml4_idx(va)];

//   // PDP: first-touch placement ??? allocate on chiplet_id so that the GMMU
//   // on chiplet_id can read this node locally during PTW.
//   uint64_t pdp_key = pt_pml4_idx(va);
//   if (m_pdp_nodes.find(pdp_key) == m_pdp_nodes.end()) {
//     pt_node_t* node = alloc_node(chiplet_id);   // ?�� 0 ?�� chiplet_id
//     assert(node != NULL);
//     m_pdp_nodes[pdp_key] = node;
//     pml4e->set_pfn(node->base_pa >> GPU_PAGE_SHIFT);
//     pml4e->set_present(true);
//     pml4e->set_writable(true);
//   }
//   gpu_pte_t* pdpe = &m_pdp_nodes[pdp_key]->entries[pt_pdp_idx(va)];

//   // PD: first-touch placement ??? same chiplet as data.
//   uint64_t pd_key = (pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
//   if (m_pd_nodes.find(pd_key) == m_pd_nodes.end()) {
//     pt_node_t* node = alloc_node(chiplet_id);   // ?�� 0 ?�� chiplet_id
//     assert(node != NULL);
//     m_pd_nodes[pd_key] = node;
//     pdpe->set_pfn(node->base_pa >> GPU_PAGE_SHIFT);
//     pdpe->set_present(true);
//     pdpe->set_writable(true);
//   }
//   gpu_pte_t* pde = &m_pd_nodes[pd_key]->entries[pt_pd_idx(va)];

//   // PT: first-touch placement ??? same chiplet as data.
//   uint64_t pt_key = ((uint64_t)pt_pml4_idx(va) << 18) |
//                     ((uint64_t)pt_pdp_idx(va)   <<  9) |
//                      (uint64_t)pt_pd_idx(va);
//   if (m_pt_nodes.find(pt_key) == m_pt_nodes.end()) {
//     pt_node_t* node = alloc_node(chiplet_id);   // ?�� 0 ?�� chiplet_id
//     assert(node != NULL);
//     m_pt_nodes[pt_key] = node;
//     pde->set_pfn(node->base_pa >> GPU_PAGE_SHIFT);
//     pde->set_present(true);
//     pde->set_writable(true);
//   }
//   gpu_pte_t* pte = &m_pt_nodes[pt_key]->entries[pt_pt_idx(va)];

//   // Leaf PTE: allocate data page from target chiplet.
//   assert(!pte->present() && "double map: VA already mapped");
//   uint64_t data_pfn = m_mod->pmm_alloc_page(chiplet_id);
//   assert(data_pfn != (uint64_t)-1 && "PMM exhausted for data page");
//   pte->set_pfn(data_pfn);
//   pte->set_present(true);
//   pte->set_writable(true);

//   printf("[PAGE_TABLE] map: va=0x%llx -> chiplet=%u "
//          "pdp_pa=0x%llx pd_pa=0x%llx pt_pa=0x%llx data_pfn=0x%llx\n",
//          (unsigned long long)va, chiplet_id,
//          (unsigned long long)m_pdp_nodes[pdp_key]->base_pa,
//          (unsigned long long)m_pd_nodes[pd_key]->base_pa,
//          (unsigned long long)m_pt_nodes[pt_key]->base_pa,
//          (unsigned long long)data_pfn);
// }

uint64_t gpu_page_table::ensure_node(uint64_t va, int level, unsigned chiplet_id) {
  // P5-2: structural change to the GPU-wide page table. Only reachable when
  // the node/leaf is absent, i.e. never under frozen placement.
  note_mutation();
  switch (level) {
    case 1: {
      uint64_t key = pt_pml4_idx(va);
      if (m_pdp_nodes.find(key) == m_pdp_nodes.end()) {
        pt_node_t* node = alloc_node(chiplet_id);
        assert(node != NULL && "ensure_node: PMM exhausted for PDP");
        m_pdp_nodes[key] = node;
        gpu_pte_t* pml4e = &m_pml4->entries[pt_pml4_idx(va)];
        pml4e->set_pfn(node->base_pa >> GPU_PAGE_SHIFT);
        pml4e->set_present(true);
        pml4e->set_writable(true);
        #ifdef LOG_PAGE_TABLE
        printf("[PAGE_TABLE] ensure_node: va=0x%llx level=1(PDP)"
               " chiplet=%u base_pa=0x%llx\n",
               (unsigned long long)va, chiplet_id,
               (unsigned long long)node->base_pa);
        #endif
      }
      return m_pdp_nodes[key]->base_pa;
    }
    case 2: {
      uint64_t key = ((uint64_t)pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
      if (m_pd_nodes.find(key) == m_pd_nodes.end()) {
        pt_node_t* node = alloc_node(chiplet_id);
        assert(node != NULL && "ensure_node: PMM exhausted for PD");
        m_pd_nodes[key] = node;
        uint64_t pdp_key = pt_pml4_idx(va);
        assert(m_pdp_nodes.count(pdp_key) && "ensure_node(2): PDP must exist first");
        gpu_pte_t* pdpe = &m_pdp_nodes[pdp_key]->entries[pt_pdp_idx(va)];
        pdpe->set_pfn(node->base_pa >> GPU_PAGE_SHIFT);
        pdpe->set_present(true);
        pdpe->set_writable(true);
        #ifdef LOG_PAGE_TABLE
        printf("[PAGE_TABLE] ensure_node: va=0x%llx level=2(PD)"
               " chiplet=%u base_pa=0x%llx\n",
               (unsigned long long)va, chiplet_id,
               (unsigned long long)node->base_pa);
        #endif
      }
      return m_pd_nodes[key]->base_pa;
    }
    case 3: {
      // A 2MB page has no level-3 node at all: its PD entry is the leaf.
      assert(m_mod->page_shift_for_va(va) != GPU_HUGE_SHIFT &&
             "ensure_node(3): a 2MB page must not create a level-3 node");
      // 4KB -> 512-entry PT, 64KB -> 32-entry big PT. Both are keyed by the PD
      // prefix and both may exist under the same PD entry.
      const bool is_64k = (m_mod->page_shift_for_va(va) == GPU_64K_SHIFT);
      auto& nodes = is_64k ? m_pt64k_nodes : m_pt_nodes;
      uint64_t key = ((uint64_t)pt_pml4_idx(va) << 18) |
                     ((uint64_t)pt_pdp_idx(va)   <<  9) |
                      (uint64_t)pt_pd_idx(va);
      if (nodes.find(key) == nodes.end()) {
        pt_node_t* node = alloc_node(chiplet_id);
        assert(node != NULL && "ensure_node: PMM exhausted for PT");
        nodes[key] = node;
        uint64_t pd_key = ((uint64_t)pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
        assert(m_pd_nodes.count(pd_key) && "ensure_node(3): PD must exist first");
        // The PDE carries a single pfn while Pascal's PDE0 has small+big pointers.
        // Harmless here: nothing traverses the PDE's pfn -- walk() and
        // get_node_base_pa() resolve nodes through the maps above, and present()
        // is correct for either child.
        gpu_pte_t* pde = &m_pd_nodes[pd_key]->entries[pt_pd_idx(va)];
        pde->set_pfn(node->base_pa >> GPU_PAGE_SHIFT);
        pde->set_present(true);
        pde->set_writable(true);
        #ifdef LOG_PAGE_TABLE
        printf("[PAGE_TABLE] ensure_node: va=0x%llx level=3(%s)"
               " chiplet=%u base_pa=0x%llx\n",
               (unsigned long long)va, (is_64k ? "PT64K" : "PT"), chiplet_id,
               (unsigned long long)node->base_pa);
        #endif
      }
      return nodes[key]->base_pa;
    }
    default:
      assert(false && "ensure_node: level must be 1, 2, or 3");
      return (uint64_t)-1;
  }
}

uint64_t gpu_page_table::ensure_node_shift(uint64_t va, int level,
                                           unsigned chiplet_id,
                                           unsigned page_shift) {
  // P5-2: structural change to the GPU-wide page table. Only reachable when
  // the node/leaf is absent, i.e. never under frozen placement.
  note_mutation();
  // Levels 1 and 2 are page-size independent.
  if (level != 3) return ensure_node(va, level, chiplet_id);
  assert(page_shift != GPU_HUGE_SHIFT &&
         "ensure_node_shift(3): a 2MB page must not create a level-3 node");

  auto& nodes = (page_shift == GPU_64K_SHIFT) ? m_pt64k_nodes : m_pt_nodes;
  uint64_t key = ((uint64_t)pt_pml4_idx(va) << 18) |
                 ((uint64_t)pt_pdp_idx(va)   <<  9) |
                  (uint64_t)pt_pd_idx(va);
  if (nodes.find(key) == nodes.end()) {
    pt_node_t* node = alloc_node(chiplet_id);
    assert(node != NULL && "ensure_node_shift: PMM exhausted for PT");
    nodes[key] = node;
    uint64_t pd_key = ((uint64_t)pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
    assert(m_pd_nodes.count(pd_key) && "ensure_node_shift(3): PD must exist first");
    gpu_pte_t* pde = &m_pd_nodes[pd_key]->entries[pt_pd_idx(va)];
    pde->set_pfn(node->base_pa >> GPU_PAGE_SHIFT);
    pde->set_present(true);
    pde->set_writable(true);
  }
  return nodes[key]->base_pa;
}

void gpu_page_table::map_leaf(uint64_t va, unsigned chiplet_id) {
  // P5-2: structural change to the GPU-wide page table. Only reachable when
  // the node/leaf is absent, i.e. never under frozen placement.
  note_mutation();
  // 2MB huge page: the PD entry is the leaf, so no PT node is created and the
  // data page needs 512 physically contiguous frames.
  if (m_mod->page_shift_for_va(va) == GPU_HUGE_SHIFT) {
    uint64_t pd_key = ((uint64_t)pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
    assert(m_pd_nodes.count(pd_key) && "map_leaf(2M): PD node must exist first");

    gpu_pte_t* pde = &m_pd_nodes[pd_key]->entries[pt_pd_idx(va)];
    assert(!pde->present() && "map_leaf(2M): double mapping attempt");

    uint64_t base_pfn = m_mod->pmm_alloc_contiguous(chiplet_id, GPU_HUGE_FRAMES);
    assert(base_pfn != (uint64_t)-1 && "map_leaf(2M): PMM exhausted for 2MB page");
    assert((base_pfn % GPU_HUGE_FRAMES) == 0 && "map_leaf(2M): base must be 2MB-aligned");

    pde->set_pfn(base_pfn);
    pde->set_present(true);
    pde->set_writable(true);
    pde->set_huge_page(true);  // PS=1: walk() terminates here

    #ifdef LOG_PAGE_TABLE
    printf("[PAGE_TABLE] map_leaf(2M): va=0x%llx chiplet=%u base_pfn=0x%llx\n",
           (unsigned long long)va, chiplet_id,
           (unsigned long long)base_pfn);
    #endif
    return;
  }

  // 64KB big page: the leaf lives in the 32-entry big PT, and the data page needs
  // 16 physically contiguous frames.
  if (m_mod->page_shift_for_va(va) == GPU_64K_SHIFT) {
    uint64_t pt64k_key = ((uint64_t)pt_pml4_idx(va) << 18) |
                         ((uint64_t)pt_pdp_idx(va)   <<  9) |
                          (uint64_t)pt_pd_idx(va);
    assert(m_pt64k_nodes.count(pt64k_key) &&
           "map_leaf(64K): big PT node must exist first");

    gpu_pte_t* pte64k = &m_pt64k_nodes[pt64k_key]->entries[pt_pt64k_idx(va)];
    assert(!pte64k->present() && "map_leaf(64K): double mapping attempt");

    uint64_t base_pfn64k = m_mod->pmm_alloc_contiguous(chiplet_id, GPU_64K_FRAMES);
    assert(base_pfn64k != (uint64_t)-1 && "map_leaf(64K): PMM exhausted for 64KB page");
    assert((base_pfn64k % GPU_64K_FRAMES) == 0 &&
           "map_leaf(64K): base must be 64KB-aligned");

    pte64k->set_pfn(base_pfn64k);
    pte64k->set_present(true);
    pte64k->set_writable(true);
    // Deliberately no set_huge_page(): that bit would make walk() stop at the PD
    // and treat this as a 2MB page. 64KB is identified by its node, not a bit.

    #ifdef LOG_PAGE_TABLE
    printf("[PAGE_TABLE] map_leaf(64K): va=0x%llx chiplet=%u base_pfn=0x%llx\n",
           (unsigned long long)va, chiplet_id,
           (unsigned long long)base_pfn64k);
    #endif
    return;
  }

  uint64_t pt_key = ((uint64_t)pt_pml4_idx(va) << 18) |
                    ((uint64_t)pt_pdp_idx(va)   <<  9) |
                     (uint64_t)pt_pd_idx(va);
  assert(m_pt_nodes.count(pt_key) && "map_leaf: PT node must exist first");

  gpu_pte_t* pte = &m_pt_nodes[pt_key]->entries[pt_pt_idx(va)];
  assert(!pte->present() && "map_leaf: double mapping attempt");

  uint64_t data_pfn = m_mod->pmm_alloc_page(chiplet_id);
  assert(data_pfn != (uint64_t)-1 && "map_leaf: PMM exhausted for data page");

  pte->set_pfn(data_pfn);
  pte->set_present(true);
  pte->set_writable(true);

  #ifdef LOG_PAGE_TABLE
  printf("[PAGE_TABLE] map_leaf: va=0x%llx chiplet=%u data_pfn=0x%llx\n",
         (unsigned long long)va, chiplet_id,
         (unsigned long long)data_pfn);
  #endif
}

bool gpu_page_table::record_first_touch(uint64_t va, unsigned chiplet) {
  uint64_t pml4 = pt_pml4_idx(va), pdp = pt_pdp_idx(va), pd = pt_pd_idx(va);
  // P5-1: emplace() is a non-const call even when the key is already present,
  // so calling it from several chiplets at once is a data race regardless of
  // whether anything is inserted. Probe with find() first: under frozen
  // placement every key exists, the whole function becomes read-only, and
  // concurrent chiplets are safe. The insert path keeps first-write-wins
  // semantics, byte for byte what emplace() did.
  // Key by the page-size-aligned VPN so one huge page has exactly one owner.
  // Must stay in sync with first_touch_data_owner().
  const uint64_t k_data = page_vpn(va, m_mod->page_shift_for_va(va));
  const uint64_t k_pd   = (pml4 << 9) | pdp;
  const uint64_t k_pt   = (pml4 << 18) | (pdp << 9) | pd;
  bool inserted = false;
  if (m_pdp_owner.find(pml4)  == m_pdp_owner.end())  { m_pdp_owner.emplace(pml4, chiplet);   note_mutation(); inserted = true; }
  if (m_pd_owner.find(k_pd)   == m_pd_owner.end())   { m_pd_owner.emplace(k_pd, chiplet);    note_mutation(); inserted = true; }
  if (m_pt_owner.find(k_pt)   == m_pt_owner.end())   { m_pt_owner.emplace(k_pt, chiplet);    note_mutation(); inserted = true; }
  if (m_data_owner.find(k_data) == m_data_owner.end()) { m_data_owner.emplace(k_data, chiplet); note_mutation(); inserted = true; }
  return inserted;
}

void gpu_page_table::note_first_touch(uint64_t va, unsigned chiplet) {
  // A1: in reference mode the insert is deferred so the parallel CORE phase
  // never mutates the table (see the header comment). Frozen-placement and
  // 1-thread non-reference runs keep the immediate path, byte for byte.
  if (m_defer_ft) {
    assert(chiplet < m_ft_buf.size());
    m_ft_buf[chiplet].push_back(va);
    return;
  }
  record_first_touch(va, chiplet);
}

void gpu_page_table::enable_deferred_first_touch(unsigned n_chiplets) {
  m_defer_ft = true;
  m_ft_buf.assign(n_chiplets, std::vector<uint64_t>());
}

void gpu_page_table::commit_deferred_first_touch() {
  if (!m_defer_ft) return;
  // Canonical order: chiplet index ascending, arrival order within a chiplet.
  // The 1-thread pass-B loop walks chiplet groups in exactly this order, so
  // committing here reproduces its owner maps bit for bit. Runs in the
  // sequential part of the cycle -- the pt-guard has already re-sampled the
  // mutation counter, and every owner-map reader (done_tlb_req, level faults)
  // executes at least one cycle after the translate that buffered the entry.
  unsigned long long batch = 0;
  for (unsigned c = 0; c < m_ft_buf.size(); c++) {
    for (uint64_t va : m_ft_buf[c]) {
      if (!record_first_touch(va, c)) m_ft_dedups++;
      m_ft_commits++;
      batch++;
    }
    m_ft_buf[c].clear();
  }
  if (batch > m_ft_max_batch) m_ft_max_batch = batch;
}

unsigned gpu_page_table::first_touch_node_owner(uint64_t va, int level) const {
  uint64_t pml4 = pt_pml4_idx(va), pdp = pt_pdp_idx(va), pd = pt_pd_idx(va);
  const std::unordered_map<uint64_t, unsigned>* m = nullptr;
  uint64_t key = 0;
  switch (level) {
    case 1: m = &m_pdp_owner; key = pml4; break;
    case 2: m = &m_pd_owner;  key = (pml4 << 9) | pdp; break;
    case 3: m = &m_pt_owner;  key = (pml4 << 18) | (pdp << 9) | pd; break;
    default: assert(false && "first_touch_node_owner: level must be 1,2,3"); return 0;
  }
  auto it = m->find(key);
  assert(it != m->end() && "first_touch_node_owner: translate() must record first");
  return it->second;
}

unsigned gpu_page_table::first_touch_data_owner(uint64_t va) const {
  // Same key rule as record_first_touch().
  auto it = m_data_owner.find(page_vpn(va, m_mod->page_shift_for_va(va)));
  assert(it != m_data_owner.end() && "first_touch_data_owner: translate() must record first");
  return it->second;
}

// ===================== placement snapshot (local-PTW experiment) =============
// Freezing placement turns first-touch into a pure placement policy: the owner
// maps are filled from a file, every page is mapped before kernel 1, and the
// measured region therefore takes zero faults.
//
// PAGE SIZE. The map is keyed by vpn = va >> page_shift, so a 4KB map and a
// 64KB map are different files that must never be crossed. The header records
// the shift and load_placement() aborts on a mismatch rather than silently
// producing a wrong experiment.

// The one page shift this configuration uses for every allocation, or 0 if the
// policy is size-dependent (mixed pages), which premap cannot replay: premap
// runs before any cudaMalloc, so allocation sizes are not known yet.
unsigned gpu_page_table::config_uniform_page_shift() const {
  return m_mod->uniform_page_shift();
}

void gpu_page_table::map_leaf_with_pfn(uint64_t va, uint64_t pfn,
                                       unsigned page_shift) {
  assert(pfn != (uint64_t)-1 && "map_leaf_with_pfn: invalid pfn");

  if (page_shift == GPU_HUGE_SHIFT) {
    // 2MB: the PD entry itself is the leaf, no level-3 node exists.
    uint64_t pd_key = ((uint64_t)pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
    assert(m_pd_nodes.count(pd_key) && "map_leaf_with_pfn(2M): PD must exist first");
    gpu_pte_t* pde = &m_pd_nodes[pd_key]->entries[pt_pd_idx(va)];
    assert(!pde->present() && "map_leaf_with_pfn(2M): double mapping attempt");
    assert((pfn % GPU_HUGE_FRAMES) == 0 && "map_leaf_with_pfn(2M): unaligned base");
    pde->set_pfn(pfn);
    pde->set_present(true);
    pde->set_writable(true);
    pde->set_huge_page(true);   // PS=1: walk() terminates here
    return;
  }

  uint64_t pt_key = ((uint64_t)pt_pml4_idx(va) << 18) |
                    ((uint64_t)pt_pdp_idx(va)   <<  9) |
                     (uint64_t)pt_pd_idx(va);

  if (page_shift == GPU_64K_SHIFT) {
    // 64KB: leaf lives in the 32-entry big PT. No set_huge_page() -- that bit
    // would make walk() stop at the PD and read this as a 2MB page.
    assert(m_pt64k_nodes.count(pt_key) &&
           "map_leaf_with_pfn(64K): big PT node must exist first");
    gpu_pte_t* pte = &m_pt64k_nodes[pt_key]->entries[pt_pt64k_idx(va)];
    assert(!pte->present() && "map_leaf_with_pfn(64K): double mapping attempt");
    assert((pfn % GPU_64K_FRAMES) == 0 && "map_leaf_with_pfn(64K): unaligned base");
    pte->set_pfn(pfn);
    pte->set_present(true);
    pte->set_writable(true);
    return;
  }

  assert(m_pt_nodes.count(pt_key) && "map_leaf_with_pfn: PT node must exist first");
  gpu_pte_t* pte = &m_pt_nodes[pt_key]->entries[pt_pt_idx(va)];
  assert(!pte->present() && "map_leaf_with_pfn: double mapping attempt");
  pte->set_pfn(pfn);
  pte->set_present(true);
  pte->set_writable(true);
}

bool gpu_page_table::dump_placement(const char* path) const {
  const unsigned shift = config_uniform_page_shift();
  if (shift == 0) {
    printf("[PLACEMENT] dump: refusing -- -gpgpu_page_size_policy is size-dependent; "
           "a placement map is only well defined for a single page size\n"); 
    return false;
  }
  FILE* f = fopen(path, "w");
  if (!f) {
    printf("[PLACEMENT] dump: cannot open %s\n", path); 
    return false;
  }
  fprintf(f, "# gpgpu-sim placement snapshot v2\n");
  fprintf(f, "# page_shift %u\n", shift);
  fprintf(f, "# D <vpn> <owner_chiplet> <data_pfn>   (data page, vpn = va >> page_shift)\n");
  fprintf(f, "# N <level> <region_key> <owner_chiplet>   (PT node, level 1=PDP 2=PD 3=PT)\n");

  // Sorted output so the file is byte-reproducible for diffing between runs.
  auto dump_owner_map = [&](const std::unordered_map<uint64_t, unsigned>& m, int level) {
    std::vector<uint64_t> keys;
    keys.reserve(m.size());
    for (const auto& kv : m) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (uint64_t k : keys)
      fprintf(f, "N %d 0x%llx %u\n", level, (unsigned long long)k, m.at(k));
  };
  dump_owner_map(m_pdp_owner, 1);
  dump_owner_map(m_pd_owner,  2);
  dump_owner_map(m_pt_owner,  3);

  std::vector<uint64_t> vpns;
  vpns.reserve(m_data_owner.size());
  for (const auto& kv : m_data_owner) vpns.push_back(kv.first);
  std::sort(vpns.begin(), vpns.end());
  size_t mapped = 0;
  for (uint64_t vpn : vpns) {
    uint64_t va = vpn << shift;   // m_data_owner is keyed by the same shift
    const gpu_pte_t* pte = walk(va);
    // A vpn can be recorded by translate() and yet never reach map_leaf() if the
    // run ended mid-walk; emit -1 so the reader can tell.
    unsigned long long pfn = (pte && pte->present())
                           ? (unsigned long long)pte->pfn() : (unsigned long long)-1;
    if (pte && pte->present()) mapped++;
    fprintf(f, "D 0x%llx %u 0x%llx\n",
            (unsigned long long)vpn, m_data_owner.at(vpn), pfn);
  }
  fclose(f);
  printf("[PLACEMENT] dump: %s  page_shift=%u data_vpns=%zu (mapped=%zu) "
         "pdp=%zu pd=%zu pt=%zu\n",
         path, shift, m_data_owner.size(), mapped,
         m_pdp_owner.size(), m_pd_owner.size(), m_pt_owner.size()); 
  return true;
}

bool gpu_page_table::load_placement(const char* path) {
  FILE* f = fopen(path, "r");
  if (!f) {
    printf("[PLACEMENT] load: cannot open %s\n", path); 
    return false;
  }
  const unsigned want = config_uniform_page_shift();
  if (want == 0) {
    printf("[PLACEMENT] load: FATAL -- -gpgpu_page_size_policy is size-dependent; "
           "premap runs before any cudaMalloc and cannot replay mixed page sizes\n"); 
    fclose(f);
    abort();
  }

  char line[256];
  size_t n_data = 0, n_node = 0;
  int file_shift = -1;
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#') {
      unsigned sh = 0;
      if (sscanf(line, "# page_shift %u", &sh) == 1) file_shift = (int)sh;
      continue;
    }
    if (line[0] == '\n') continue;
    if (line[0] == 'D') {
      unsigned long long vpn = 0, pfn = 0;
      unsigned owner = 0;
      if (sscanf(line, "D 0x%llx %u 0x%llx", &vpn, &owner, &pfn) != 3) continue;
      // Only the owner is replayed; the PFN in the file is for auditing. The
      // measurement run re-derives PFNs deterministically in premap_from_placement().
      m_data_owner.emplace((uint64_t)vpn, owner);
      n_data++;
    } else if (line[0] == 'N') {
      int level = 0;
      unsigned long long key = 0;
      unsigned owner = 0;
      if (sscanf(line, "N %d 0x%llx %u", &level, &key, &owner) != 3) continue;
      if      (level == 1) m_pdp_owner.emplace((uint64_t)key, owner);
      else if (level == 2) m_pd_owner.emplace((uint64_t)key, owner);
      else if (level == 3) m_pt_owner.emplace((uint64_t)key, owner);
      else continue;
      n_node++;
    }
  }
  fclose(f);

  // Hard-fail, never warn-and-continue: a vpn means a different VA under a
  // different shift, so a crossed map silently produces a bogus experiment.
  if (file_shift < 0) {
    printf("[PLACEMENT] load: FATAL -- %s has no '# page_shift' header "
           "(v1 map from a 4KB-only build?). Regenerate the reference run.\n",
           path); 
    abort();
  }
  if ((unsigned)file_shift != want) {
    printf("[PLACEMENT] load: FATAL -- %s was dumped with page_shift=%d but this "
           "config uses page_shift=%u. A placement map is page-size specific.\n",
           path, file_shift, want); 
    abort();
  }

  m_placement_page_shift = file_shift;
  m_placement_frozen = true;
  printf("[PLACEMENT] load: %s  page_shift=%d data_vpns=%zu nodes=%zu (placement frozen)\n",
         path, file_shift, n_data, n_node); 
  return true;
}

void gpu_page_table::premap_from_placement() {
  assert(m_placement_frozen && "premap_from_placement: load_placement() first");
  assert(m_placement_page_shift > 0);
  const unsigned shift = (unsigned)m_placement_page_shift;
  // Frames per data page: 1 for 4KB, 16 for 64KB, 512 for 2MB. A PT NODE page is
  // always 4KB regardless -- only the data page scales.
  const uint64_t frames = 1ULL << (shift - GPU_PAGE_SHIFT);

  std::vector<uint64_t> vpns;
  vpns.reserve(m_data_owner.size());
  for (const auto& kv : m_data_owner) vpns.push_back(kv.first);
  std::sort(vpns.begin(), vpns.end());

  // (A) Reserve every data-page PFN first, in ascending vpn order. Doing this
  // before any PT node is allocated is what makes the data-page PFNs identical
  // between the baseline and local-PTW runs (the latter allocates n_chiplets x
  // the PT node pages, which would otherwise shift every subsequent PFN).
  std::vector<uint64_t> data_pfn(vpns.size());
  for (size_t i = 0; i < vpns.size(); i++) {
    unsigned owner = m_data_owner.at(vpns[i]);
    // 64KB/2MB data pages need physically contiguous, naturally aligned frames.
    data_pfn[i] = (frames == 1) ? m_mod->pmm_alloc_page(owner)
                                : m_mod->pmm_alloc_contiguous(owner, frames);
    assert(data_pfn[i] != (uint64_t)-1 && "premap: PMM exhausted for data page");
  }

  // Only now may the root's per-chiplet copies consume pages. Doing it earlier
  // (e.g. in the constructor) would take one page from each chiplet pool ahead
  // of phase (A), shifting every data-page PFN relative to the baseline run.
  alloc_root_replicas();

  // (B) PT node chains, then (C) leaf PTEs pointing at the reserved PFNs.
  // ensure_node()/map_leaf() consult page_shift_for_va(), which cannot work here
  // (no cudaMalloc yet), so level 3 is selected from `shift` directly and the
  // leaf is installed by the shift-aware map_leaf_with_pfn().
  for (size_t i = 0; i < vpns.size(); i++) {
    uint64_t va = vpns[i] << shift;
    ensure_node(va, 1, first_touch_node_owner(va, 1));
    ensure_node(va, 2, first_touch_node_owner(va, 2));
    if (shift != GPU_HUGE_SHIFT)   // 2MB has no level-3 node: the PD is the leaf
      ensure_node_shift(va, 3, first_touch_node_owner(va, 3), shift);
    map_leaf_with_pfn(va, data_pfn[i], shift);
  }

  printf("[PLACEMENT] premap: mapped %zu data pages (page_shift=%u), "
         "pt_nodes=%zu pt_pages=%zu (%.2f MB)\n",
         vpns.size(), shift, pt_node_count(), pt_page_count(),
         (double)(pt_page_count() * GPU_PAGE_SIZE) / (1024.0 * 1024.0)); 
}
// ============================================================================

void gpu_page_table::unmap(uint64_t va) {
  // Walk to the leaf PTE and clear present bit.
  // Does not free the physical page or any PT node (Phase 3 cudaFree hook will handle that).
  uint64_t pt_key = ((uint64_t)pt_pml4_idx(va) << 18) |
                    ((uint64_t)pt_pdp_idx(va)   <<  9) |
                     (uint64_t)pt_pd_idx(va);
  auto pt_it = m_pt_nodes.find(pt_key);
  if (pt_it == m_pt_nodes.end()) return;  // not mapped, nothing to do

  gpu_pte_t* pte = &pt_it->second->entries[pt_pt_idx(va)];
  if (pte->present()) {
    pte->clear();
    printf("[PAGE_TABLE] unmap: va=0x%llx cleared\n", (unsigned long long)va);
  }
}

void gpu_page_table::dump(FILE* f) const {
  if (!f) f = stdout;
  fprintf(f, "==== gpu_page_table dump ====\n");
  if (!m_pml4) { fprintf(f, "  (empty)\n"); return; }
  fprintf(f, "  PML4 base_pa=0x%llx\n", (unsigned long long)m_pml4->base_pa);
  fprintf(f, "  PDP nodes : %zu\n", m_pdp_nodes.size());
  fprintf(f, "  PD  nodes : %zu\n", m_pd_nodes.size());
  fprintf(f, "  PT  nodes : %zu\n", m_pt_nodes.size());
  fprintf(f, "============================\n");
}

uint64_t gpu_page_table::get_node_base_pa(uint64_t va, int level) const {
  switch (level) {
    case 0:
      return m_pml4 ? m_pml4->base_pa : (uint64_t)-1;
    case 1: {
      uint64_t key = pt_pml4_idx(va);
      auto it = m_pdp_nodes.find(key);
      return it != m_pdp_nodes.end() ? it->second->base_pa : (uint64_t)-1;
    }
    case 2: {
      uint64_t key = ((uint64_t)pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
      auto it = m_pd_nodes.find(key);
      return it != m_pd_nodes.end() ? it->second->base_pa : (uint64_t)-1;
    }
    case 3: {
      uint64_t key = ((uint64_t)pt_pml4_idx(va) << 18)
                   | ((uint64_t)pt_pdp_idx(va)   <<  9)
                   | pt_pd_idx(va);
      // 64KB reads its own big PT node; 4KB reads the 512-entry PT. The GMMU uses
      // this to address the level-3 DRAM read, so it must agree with walk().
      const auto& nodes = (m_mod->page_shift_for_va(va) == GPU_64K_SHIFT)
                        ? m_pt64k_nodes : m_pt_nodes;
      auto it = nodes.find(key);
      return it != nodes.end() ? it->second->base_pa : (uint64_t)-1;
    }
    default:
      return (uint64_t)-1;
  }
}

// Per-chiplet variant. MUST mirror the 2-arg form's page-size dispatch exactly;
// if level 3 picked the wrong map, a 64KB walk would read a 4KB node's address.
uint64_t gpu_page_table::get_node_base_pa(uint64_t va, int level,
                                          unsigned chiplet) const {
  switch (level) {
    case 0:
      return m_pml4 ? m_pml4->base_pa_for(chiplet) : (uint64_t)-1;
    case 1: {
      uint64_t key = pt_pml4_idx(va);
      auto it = m_pdp_nodes.find(key);
      return it != m_pdp_nodes.end() ? it->second->base_pa_for(chiplet)
                                     : (uint64_t)-1;
    }
    case 2: {
      uint64_t key = ((uint64_t)pt_pml4_idx(va) << 9) | pt_pdp_idx(va);
      auto it = m_pd_nodes.find(key);
      return it != m_pd_nodes.end() ? it->second->base_pa_for(chiplet)
                                    : (uint64_t)-1;
    }
    case 3: {
      uint64_t key = ((uint64_t)pt_pml4_idx(va) << 18)
                   | ((uint64_t)pt_pdp_idx(va)   <<  9)
                   | pt_pd_idx(va);
      // 2MB never reaches level 3 (its leaf is the PD), so -1 here is correct
      // for it, same as the 2-arg form.
      const auto& nodes = (m_mod->page_shift_for_va(va) == GPU_64K_SHIFT)
                        ? m_pt64k_nodes : m_pt_nodes;
      auto it = nodes.find(key);
      return it != nodes.end() ? it->second->base_pa_for(chiplet)
                               : (uint64_t)-1;
    }
    default:
      return (uint64_t)-1;
  }
}

// P3
void gpu_module::on_malloc(uint64_t va_base, size_t aligned_size,
                           unsigned page_shift) {
  // A trace can carry several MemcpyHtoD into the same buffer; registering it
  // twice would leave two overlapping ranges and make page_shift_for_va()
  // depend on insertion order.
  if (is_va_allocated(va_base)) return;
  m_va_alloc_list.push_back({va_base, aligned_size, page_shift});
  printf("[MALLOC_HOOK] va=0x%llx aligned_size=%zu page=%lluKB\n",
         (unsigned long long)va_base, aligned_size,
         (unsigned long long)((1ULL << page_shift) >> 10));
}

// P3
bool gpu_module::is_va_allocated(uint64_t va) const {
  for (const auto& a : m_va_alloc_list) {
    if (va >= a.va_base && va < a.va_base + (uint64_t)a.aligned_size)
      return true;
  }
  return false;
}

// Linear scan is fine here: a benchmark registers on the order of tens of
// allocations. If that ever grows into the thousands, switch m_va_alloc_list to
// a std::map keyed by va_base and use upper_bound.
unsigned gpu_module::page_shift_for_va(uint64_t va) const {
  for (const auto& a : m_va_alloc_list) {
    if (va >= a.va_base && va < a.va_base + (uint64_t)a.aligned_size)
      return a.page_shift;
  }
  // Unregistered VA (local memory, kernel params, an output buffer that never
  // got a MemcpyHtoD, ...). Under a FIXED policy the config states that every
  // allocation has one size, so answering 4KB here would manufacture a mixed
  // page-size run that nobody asked for -- inputs 64KB, outputs 4KB. Answer the
  // configured size instead. uniform_page_shift() returns 0 for the automatic
  // policies (3-5), where the size is genuinely unknown, and 4KB stays correct.
  const unsigned uniform = uniform_page_shift();
  return uniform ? uniform : GPU_PAGE_SHIFT;
}

// P3
gmmu* gpu_module::get_gmmu(unsigned chiplet_id) const {
  assert(chiplet_id < m_gmmus.size() && "GMMU not initialized yet");
  return m_gmmus[chiplet_id];
}

#ifdef ADDR_XLAT_VERIFY
// ===========================================================================
// Address Translation Verification harness.
// Every step is a pure checking function: asserts invariants and prints one
// [XLAT_VERIFY] ... PASS line. Kept isolated in this single block at the end
// of the file so production code stays untouched.
// ===========================================================================

// (step functions accumulate here)

// Build a canonical VA from 9-bit level indices (x4) + 12-bit offset.
// Inverse of pt_*_idx().
static uint64_t build_va(unsigned pml4, unsigned pdp, unsigned pd,
                         unsigned pt, uint64_t off) {
  return ((uint64_t)(pml4 & 0x1FF) << 39) |
         ((uint64_t)(pdp  & 0x1FF) << 30) |
         ((uint64_t)(pd   & 0x1FF) << 21) |
         ((uint64_t)(pt   & 0x1FF) << 12) |
         (off & 0xFFF);
}

// V-PT-1: VA index decomposition correctness.
static void verify_va_index_decode() {
  printf("[XLAT_VERIFY V-PT-1] --- VA index decomposition ---\n");

  // (1) distinct pattern -- expected values are literals (independent oracle)
  {
    uint64_t va = build_va(1, 2, 3, 4, 0xABC);
    assert(pt_pml4_idx(va) == 1     && "V-PT-1: pml4 idx");
    assert(pt_pdp_idx(va)  == 2     && "V-PT-1: pdp idx");
    assert(pt_pd_idx(va)   == 3     && "V-PT-1: pd idx");
    assert(pt_pt_idx(va)   == 4     && "V-PT-1: pt idx");
    assert(pt_offset(va)   == 0xABC && "V-PT-1: offset");
    assert(pt_vpn(va)      == (va >> 12) && "V-PT-1: vpn");
  }
  // (2) all-zero
  {
    uint64_t va = build_va(0, 0, 0, 0, 0);
    assert(pt_pml4_idx(va)==0 && pt_pdp_idx(va)==0 &&
           pt_pd_idx(va)==0  && pt_pt_idx(va)==0  &&
           pt_offset(va)==0  && "V-PT-1: all-zero");
  }
  // (3) all-max + offset 0xFFF
  {
    uint64_t va = build_va(511, 511, 511, 511, 0xFFF);
    assert(pt_pml4_idx(va)==511 && pt_pdp_idx(va)==511 &&
           pt_pd_idx(va)==511  && pt_pt_idx(va)==511  &&
           pt_offset(va)==0xFFF && "V-PT-1: all-max");
  }
  // (4) field independence (no bleed into adjacent fields)
  {
    assert(pt_pml4_idx(build_va(0x1FF,0,0,0,0))==0x1FF && "V-PT-1: pml4 set");
    assert(pt_pdp_idx (build_va(0x1FF,0,0,0,0))==0     && "V-PT-1: pml4->pdp bleed");
    assert(pt_pdp_idx (build_va(0,0x1FF,0,0,0))==0x1FF && "V-PT-1: pdp set");
    assert(pt_pd_idx  (build_va(0,0x1FF,0,0,0))==0     && "V-PT-1: pdp->pd bleed");
    assert(pt_pd_idx  (build_va(0,0,0x1FF,0,0))==0x1FF && "V-PT-1: pd set");
    assert(pt_pt_idx  (build_va(0,0,0x1FF,0,0))==0     && "V-PT-1: pd->pt bleed");
    assert(pt_pt_idx  (build_va(0,0,0,0x1FF,0))==0x1FF && "V-PT-1: pt set");
    assert(pt_offset  (build_va(0,0,0,0x1FF,0))==0     && "V-PT-1: pt->offset bleed");
  }
  // (5) round-trip (auxiliary: build_va used both directions)
  {
    const unsigned pat[][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1},
                               {5,17,400,300},{511,1,256,8}};
    for (const auto& p : pat) {
      uint64_t va = build_va(p[0],p[1],p[2],p[3], 0x123);
      uint64_t rebuilt = build_va(pt_pml4_idx(va), pt_pdp_idx(va),
                                  pt_pd_idx(va), pt_pt_idx(va),
                                  pt_offset(va));
      assert(rebuilt == va && "V-PT-1: round-trip");
    }
  }

  printf("[XLAT_VERIFY V-PT-1] all index patterns OK : PASS\n");
}

// V-PT-3: PTE bit-field correctness (gpu_pte_t).
// Oracle = the bit-layout spec itself (explicit raw literals), independent of
// the struct's own getters/setters.
static void verify_pte_bitfields() {
  printf("[XLAT_VERIFY V-PT-3] --- PTE bit-fields ---\n");

  // (1) bit positions: drive raw with a single bit, expect only that reader true
  {
    gpu_pte_t p;
    p.raw = (1ULL << 0); assert( p.present()   && "V-PT-3: bit0=present");
    p.raw = (1ULL << 1); assert( p.writable()  && !p.present() && "V-PT-3: bit1=writable");
    p.raw = (1ULL << 5); assert( p.accessed()  && "V-PT-3: bit5=accessed");
    p.raw = (1ULL << 6); assert( p.dirty()     && "V-PT-3: bit6=dirty");
    p.raw = (1ULL << 7); assert( p.huge_page() && "V-PT-3: bit7=PS/huge");
  }
  // (2) PFN sits at bits[51:12], nowhere else
  {
    gpu_pte_t p; p.raw = 0;
    p.set_pfn(0xABCDEULL);
    assert(p.pfn() == 0xABCDEULL         && "V-PT-3: pfn roundtrip");
    assert(p.pa()  == (0xABCDEULL << 12) && "V-PT-3: pa = pfn<<12");
    assert(p.raw   == (0xABCDEULL << 12) && "V-PT-3: pfn raw placement");
  }
  // (3) PFN truncated to 40 bits (upper bits dropped)
  {
    gpu_pte_t p; p.raw = 0;
    p.set_pfn(0xFFFFFFFFFFFULL);                 // 44 bits set
    assert(p.pfn() == 0xFFFFFFFFFFULL && "V-PT-3: pfn 40-bit mask");  // 40 ones
  }
  // (4) flag set/clear round-trip + independence (flags vs PFN must not interfere)
  {
    gpu_pte_t p; p.clear();
    p.set_pfn(0x12345ULL);
    p.set_present(true);
    assert(p.present() && p.pfn()==0x12345ULL && "V-PT-3: present keeps pfn");
    assert(!p.writable() && !p.accessed() && !p.dirty() && "V-PT-3: present-only");
    p.set_writable(true); p.set_accessed(true); p.set_dirty(true);
    assert(p.writable() && p.accessed() && p.dirty() && p.present() &&
           p.pfn()==0x12345ULL && "V-PT-3: all flags + pfn coexist");
    p.set_present(false);
    assert(!p.present() && p.dirty() && p.pfn()==0x12345ULL && "V-PT-3: clear present keeps rest");
    p.set_pfn(0x6789AULL);                       // must not disturb flags
    assert(p.dirty() && p.writable() && p.pfn()==0x6789AULL && "V-PT-3: set_pfn keeps flags");
  }
  // (5) clear() zeroes all; valid()==present()
  {
    gpu_pte_t p; p.raw = ~0ULL;
    p.clear();
    assert(p.raw==0 && !p.present() && !p.dirty() && p.pfn()==0 && "V-PT-3: clear");
    p.set_present(true);
    assert(p.valid()==p.present() && "V-PT-3: valid==present");
  }

  printf("[XLAT_VERIFY V-PT-3] all PTE bit-field cases OK : PASS\n");
}

// V-PT-2: page-table build via the REAL demand-paging path (ensure_node x3 +
// map_leaf), exactly as gmmu::done_tlb_req does per fault, but synchronous here.
// map_va()/map() (eager) is the dead path and is intentionally NOT used.
// VA region pml4=256 cannot collide with real cudaMalloc (pml4=1 upward).
// unmap() does not free pages -> permanently consumes 10 pages on chiplet 0
// (negligible); keep ADDR_XLAT_VERIFY OFF for measurement runs.
static void verify_pt_demand_build(gpu_module* mod) {
  printf("[XLAT_VERIFY V-PT-2] --- demand build (ensure_node + map_leaf) ---\n");

  const unsigned C = 0;          // place everything on chiplet 0
  const unsigned P = 256;        // pml4 index, collision-free
  chiplet*        chip0 = mod->get_chiplet(C);
  gpu_page_table* pt    = mod->get_page_table();

  uint64_t va1 = build_va(P, 0, 0, 0, 0x10);  // fresh subtree    -> 4
  uint64_t va2 = build_va(P, 1, 0, 0, 0x20);  // PDP reused       -> 3
  uint64_t va3 = build_va(P, 0, 1, 0, 0x30);  // PDP+PD reused    -> 2
  uint64_t va4 = build_va(P, 0, 0, 1, 0x40);  // PDP+PD+PT reused -> 1

  // ---- va1: fresh subtree (order mirrors done_tlb_req: L1->L2->L3->leaf) ----
  size_t f0 = chip0->free_page_count();
  uint64_t pdp1 = pt->ensure_node(va1, 1, C);
  uint64_t pd1  = pt->ensure_node(va1, 2, C);
  uint64_t pt1  = pt->ensure_node(va1, 3, C);
  pt->map_leaf(va1, C);
  size_t f1 = chip0->free_page_count();
  assert(f0 - f1 == 4 && "V-PT-2: va1 PDP+PD+PT+data");

  // ---- va2: shares PDP only ----
  uint64_t pdp2 = pt->ensure_node(va2, 1, C);
  pt->ensure_node(va2, 2, C);
  pt->ensure_node(va2, 3, C);
  pt->map_leaf(va2, C);
  size_t f2 = chip0->free_page_count();
  assert(f1 - f2 == 3 && "V-PT-2: va2 reuses PDP");
  assert(pdp2 == pdp1 && "V-PT-2: PDP node reused (same base_pa)");

  // ---- va3: shares PDP + PD ----
  uint64_t pdp3 = pt->ensure_node(va3, 1, C);
  uint64_t pd3  = pt->ensure_node(va3, 2, C);
  pt->ensure_node(va3, 3, C);
  pt->map_leaf(va3, C);
  size_t f3 = chip0->free_page_count();
  assert(f2 - f3 == 2 && "V-PT-2: va3 reuses PDP+PD");
  assert(pdp3 == pdp1 && pd3 == pd1 && "V-PT-2: PDP+PD reused");

  // ---- va4: shares PDP + PD + PT ----
  uint64_t pdp4 = pt->ensure_node(va4, 1, C);
  uint64_t pd4  = pt->ensure_node(va4, 2, C);
  uint64_t pt4  = pt->ensure_node(va4, 3, C);
  pt->map_leaf(va4, C);
  size_t f4 = chip0->free_page_count();
  assert(f3 - f4 == 1 && "V-PT-2: va4 reuses PDP+PD+PT");
  assert(pdp4 == pdp1 && pd4 == pd1 && pt4 == pt1 && "V-PT-2: PDP+PD+PT reused");
  assert(f0 - f4 == 10 && "V-PT-2: total 10 (PT-A conservation)");

  // walk returns present leaf at depth 4 for every faulted-in VA
  int d1=0, d2=0, d3=0, d4=0;
  const gpu_pte_t* w1 = mod->walk_va(va1, &d1);
  const gpu_pte_t* w2 = mod->walk_va(va2, &d2);
  const gpu_pte_t* w3 = mod->walk_va(va3, &d3);
  const gpu_pte_t* w4 = mod->walk_va(va4, &d4);
  assert(w1 && w1->present() && d1==4 && "V-PT-2: va1 walk");
  assert(w2 && w2->present() && d2==4 && "V-PT-2: va2 walk");
  assert(w3 && w3->present() && d3==4 && "V-PT-2: va3 walk");
  assert(w4 && w4->present() && d4==4 && "V-PT-2: va4 walk");

  // 4 distinct data pages -> 4 distinct PFNs (structural oracle)
  assert(w1->pfn()!=w2->pfn() && w1->pfn()!=w3->pfn() && w1->pfn()!=w4->pfn() &&
         w2->pfn()!=w3->pfn() && w2->pfn()!=w4->pfn() && w3->pfn()!=w4->pfn() &&
         "V-PT-2: distinct data PFNs");

  // unmap clears only the target leaf; siblings stay mapped
  pt->unmap(va1);
  assert(mod->walk_va(va1)==nullptr && "V-PT-2: va1 unmapped");
  assert(mod->walk_va(va2)!=nullptr && "V-PT-2: va2 still mapped");
  assert(mod->walk_va(va3)!=nullptr && "V-PT-2: va3 still mapped");
  assert(mod->walk_va(va4)!=nullptr && "V-PT-2: va4 still mapped");

  printf("[XLAT_VERIFY V-PT-2] demand build + reuse + conservation OK : PASS\n");
}

// V-PWC: Page Walk Cache key / lookup / insert / LRU correctness.
// Uses throwaway gmmu instances (small PWC) so live GMMUs are untouched.
// pwc_lookup() is const and does NOT refresh lru_time, so eviction is by
// insertion/update time -- verified here as the code actually behaves.
static void verify_pwc(gpu_module* mod) {
  printf("[XLAT_VERIFY V-PWC] --- PWC key / lookup / insert / LRU ---\n");

  const unsigned CAP = 4;
  std::vector<unsigned>   dummy_c2c;   // unused by PWC primitives
  std::vector<va_alloc_t> dummy_va;    // unused by PWC primitives

  // (1) key correctness vs independent formula literals
  {
    gmmu g(0, 1, 8, CAP, mod->get_page_table(), &dummy_c2c, &dummy_va, mod);
    assert(g.xlat_pwc_capacity()==CAP && "V-PWC: capacity");
    uint64_t va = build_va(7, 3, 5, 9, 0);
    assert(g.xlat_pwc_key(va,0) == 7ULL                         && "V-PWC: L0 key=pml4");
    assert(g.xlat_pwc_key(va,1) == ((7ULL<<9)|3)                && "V-PWC: L1 key");
    assert(g.xlat_pwc_key(va,2) == ((7ULL<<18)|(3ULL<<9)|5)     && "V-PWC: L2 key");
    assert(g.xlat_pwc_key(va,3) == (uint64_t)-1                 && "V-PWC: leaf not cached");
  }

  // (2) insert/lookup hit; prefix-sharing + level isolation
  {
    gmmu g(0, 1, 8, CAP, mod->get_page_table(), &dummy_c2c, &dummy_va, mod);
    uint64_t vA = build_va(7,3,5,0,0);   // L2 prefix (7,3,5)
    uint64_t vB = build_va(7,3,5,1,0);   // same L2 prefix, different leaf
    uint64_t vC = build_va(7,3,6,0,0);   // different pd -> different L2 key
    uint64_t out = 0;
    g.xlat_pwc_insert(vA, 2, 0xDEAD000ULL, 100);
    assert(g.xlat_pwc_lookup(vB,2,out) && out==0xDEAD000ULL && "V-PWC: shared prefix hit");
    assert(!g.xlat_pwc_lookup(vC,2,out)                     && "V-PWC: different prefix miss");
    assert(!g.xlat_pwc_lookup(vA,1,out)                     && "V-PWC: other level empty");
  }

  // (3) LRU eviction by insertion time (CAP+1 distinct L2 keys)
  {
    gmmu g(0, 1, 8, CAP, mod->get_page_table(), &dummy_c2c, &dummy_va, mod);
    uint64_t out = 0;
    uint64_t vs[CAP+1];
    for (unsigned i=0;i<CAP+1;i++) vs[i] = build_va(1, 0, i, 0, 0);  // distinct pd -> distinct L2 key
    for (unsigned i=0;i<CAP+1;i++) g.xlat_pwc_insert(vs[i], 2, 0x1000ULL+i, (uint64_t)i);
    assert(!g.xlat_pwc_lookup(vs[0],2,out) && "V-PWC: LRU evicted oldest");
    for (unsigned i=1;i<CAP+1;i++)
      assert(g.xlat_pwc_lookup(vs[i],2,out) && out==0x1000ULL+i && "V-PWC: survivors hit");
  }

  // (4) re-insert same key updates value, not a new slot
  {
    gmmu g(0, 1, 8, CAP, mod->get_page_table(), &dummy_c2c, &dummy_va, mod);
    uint64_t out = 0;
    uint64_t v0 = build_va(1,0,0,0,0);
    g.xlat_pwc_insert(v0, 2, 0xAAAULL, 1);
    g.xlat_pwc_insert(v0, 2, 0xBBBULL, 2);   // same key -> update
    assert(g.xlat_pwc_lookup(v0,2,out) && out==0xBBBULL && "V-PWC: update existing value");
  }

  printf("[XLAT_VERIFY V-PWC] key/lookup/insert/LRU OK : PASS\n");
}

// V-L1: L1 TLB fully-assoc/LRU + 10-cycle lookup gate + MSHR entries/max_merge.
// Standalone instances with controlled cycles; independent of live SMs.
static void verify_l1_tlb() {
  printf("[XLAT_VERIFY V-L1] --- L1 TLB FA/LRU/latency/MSHR/merge ---\n");
  const unsigned SETS=1, ASSOC=32, MSHR_E=32, MAXMERGE=192, LAT=10;

  // (1) lookup latency gate: a cached hit still takes exactly LAT cycles.
  //     New model: access() injects into the pipe; cycle() advances it; resolve
  //     happens at the pipe head after LAT cycle() calls.
  {
    gpu_l1_tlb tlb(SETS, ASSOC, MSHR_E, MAXMERGE, LAT, /*sm_id=*/0, GPU_PAGE_SHIFT);
    gpu_pte_t p; p.raw=0; p.set_pfn(0xABCULL); p.set_present(true);
    tlb.fill(7, p, 0);                       // VPN 7 cached
    uint64_t va = 7ULL << GPU_PAGE_SHIFT; gpu_pte_t out;
    assert(tlb.access(va,0,out,100)==gpu_l1_tlb::LOOKUP_PENDING && "V-L1: pending @inject");
    for (unsigned c=1;c<LAT;c++) {           // probing: still pending
      tlb.cycle(100+c);
      assert(tlb.access(va,0,out,100+c)==gpu_l1_tlb::LOOKUP_PENDING && "V-L1: pending while probing");
    }
    tlb.cycle(100+LAT);                      // LAT-th cycle resolves the pipe head
    assert(tlb.access(va,0,out,100+LAT)==gpu_l1_tlb::HIT && out.pfn()==0xABCULL && "V-L1: hit after LAT cycles");
  }

  // (2) fully-associative + 32-entry capacity + LRU (all VPNs map to set 0).
  //     New model: an uncached (evicted) VPN resolves to MSHR_HIT (allocated /
  //     in-flight), not MISS; cached survivors resolve to HIT.
  {
    gpu_l1_tlb tlb(SETS, ASSOC, MSHR_E, MAXMERGE, LAT, 0, GPU_PAGE_SHIFT);
    for (unsigned i=0;i<ASSOC+1;i++){              // 33 fills into a 32-way single set
      gpu_pte_t p; p.raw=0; p.set_pfn(0x100ULL+i); p.set_present(true);
      tlb.fill(i, p, (uint64_t)i);                 // lru_time = i
    }
    gpu_pte_t out;
    // resolve helper: inject, advance the pipe LAT cycles, then read the result.
    auto RES = [&](uint64_t vpn_, uint64_t t0) -> gpu_l1_tlb::result_t {
      uint64_t va = vpn_ << GPU_PAGE_SHIFT;
      tlb.access(va,0,out,t0);
      for (unsigned c=1;c<=LAT;c++) tlb.cycle(t0+c);
      return tlb.access(va,0,out,t0+LAT);
    };
    assert(RES(0,1000)==gpu_l1_tlb::MSHR_HIT && "V-L1: LRU evicted oldest (VPN0 -> in-flight)");
    for (unsigned i=1;i<=ASSOC;i++)
      assert(RES(i, 1000+i*100)==gpu_l1_tlb::HIT && "V-L1: survivors hit");
  }

  // (3) MSHR max_merge: one VPN accepts up to MAXMERGE merges
  {
    l1_tlb_mshr mshr(MSHR_E, MAXMERGE);
    uint64_t v=0xBEEFULL;
    for (unsigned i=0;i<MAXMERGE;i++){ assert(!mshr.full(v) && "V-L1: under merge cap"); mshr.add(v, i); }
    assert(mshr.full(v) && "V-L1: VPN full at max_merge");   // 193rd blocked
  }

  // (4) MSHR table capacity: MSHR_E distinct VPNs, then a new VPN is full
  {
    l1_tlb_mshr mshr(MSHR_E, MAXMERGE);
    for (unsigned i=0;i<MSHR_E;i++){ assert(!mshr.full(1000+i)); mshr.add(1000+i, 0); }
    assert(mshr.full(9999) && "V-L1: MSHR table full at MSHR_E entries");
  }

  // (5) overflow recovery: a full MSHR frees a slot (evict) and accepts again
  {
    l1_tlb_mshr mshr(MSHR_E, MAXMERGE);
    for (unsigned i=0;i<MSHR_E;i++) mshr.add(1000+i, 0);
    assert(mshr.full(9999) && "V-L1: full before recovery");
    mshr.evict(1000);                        // a fill+tick frees one entry
    assert(!mshr.full(9999) && "V-L1: not full after evict");
    mshr.add(9999, 0);                          // new VPN now accepted (no crash)
  }

  // (6) one logical miss is determined/counted exactly once across retries.
  //     New model: miss resolves to MSHR_HIT (allocated/in-flight); re-polls
  //     (without further cycle()) stay MSHR_HIT with no re-gate / re-count.
  {
    gpu_l1_tlb tlb(SETS, ASSOC, MSHR_E, MAXMERGE, LAT, 0, GPU_PAGE_SHIFT);
    uint64_t va = 0x5000ULL << GPU_PAGE_SHIFT; gpu_pte_t out;
    assert(tlb.access(va,0,out,0)==gpu_l1_tlb::LOOKUP_PENDING && "V-L1: pending @inject");
    for (unsigned c=1;c<LAT;c++) {
      tlb.cycle(c);
      assert(tlb.access(va,0,out,c)==gpu_l1_tlb::LOOKUP_PENDING && "V-L1: gating");
    }
    tlb.cycle(LAT);
    assert(tlb.access(va,0,out,LAT)==gpu_l1_tlb::MSHR_HIT && "V-L1: miss -> MSHR allocated (in-flight)");
    for (unsigned k=1;k<=50;k++)              // re-poll without cycle(): stays in-flight, no re-gate
      assert(tlb.access(va,0,out,LAT+k)==gpu_l1_tlb::MSHR_HIT && "V-L1: stays in-flight (no re-gate)");
    assert(tlb.xlat_misses()==1 && "V-L1: miss counted exactly once");
  }

  printf("[XLAT_VERIFY V-L1] FA/LRU/latency/MSHR/merge OK : PASS\n");
}

// V-L2: L2 TLB set-assoc/LRU + 80-cycle lookup gate + MSHR table/merge + overflow recovery.
// Standalone instances with controlled cycles; independent of live chiplets.
static void verify_l2_tlb() {
  printf("[XLAT_VERIFY V-L2] --- L2 TLB set-assoc/LRU/latency/MSHR/merge/overflow ---\n");
  const unsigned SETS=128, ASSOC=8, MSHR_E=128, MAXMERGE=46, LAT=80, CH=0;
  gpu_pte_t out;

  // (1) lookup latency gate: a cached hit takes exactly LAT cycles
  {
    gpu_l2_tlb t(SETS, ASSOC, MSHR_E, MAXMERGE, LAT, CH, GPU_PAGE_SHIFT);
    gpu_pte_t p; p.raw=0; p.set_pfn(0xABCULL); p.set_present(true);
    t.fill(7, p, 0);
    uint64_t va = 7ULL << GPU_PAGE_SHIFT;
    assert(t.access(va,9,out,100)==gpu_l2_tlb::LOOKUP_PENDING && "V-L2: hit pending @t");
    assert(t.access(va,9,out,179)==gpu_l2_tlb::LOOKUP_PENDING && "V-L2: pending @t+79");
    assert(t.access(va,9,out,180)==gpu_l2_tlb::HIT && out.pfn()==0xABCULL && "V-L2: hit @t+80");
  }

  // (2) 8-way set-assoc + LRU within a set; other sets independent
  {
    gpu_l2_tlb t(SETS, ASSOC, MSHR_E, MAXMERGE, LAT, CH, GPU_PAGE_SHIFT);
    for (unsigned i=0;i<ASSOC+1;i++){          // 9 fills, all into set 0 (vpn = i*SETS)
      gpu_pte_t p; p.raw=0; p.set_pfn(0x200ULL+i); p.set_present(true);
      t.fill((uint64_t)i*SETS, p, (uint64_t)i);
    }
    #define L2_RES(T,VPN) ( (void)t.access((uint64_t)(VPN)<<GPU_PAGE_SHIFT,0,out,(T)), \
                            t.access((uint64_t)(VPN)<<GPU_PAGE_SHIFT,0,out,(T)+LAT) )
    assert(L2_RES(1000, 0)==gpu_l2_tlb::MISS && "V-L2: 8-way evicts oldest");
    for (unsigned i=1;i<=ASSOC;i++)
      assert(L2_RES(2000+i*200, i*SETS)==gpu_l2_tlb::HIT && "V-L2: set survivors hit");
    gpu_pte_t p; p.raw=0; p.set_pfn(0x999ULL); p.set_present(true);
    t.fill(5, p, 500);                          // a different set
    assert(L2_RES(4000, 5)==gpu_l2_tlb::HIT && "V-L2: other set independent");
    #undef L2_RES
  }

  // (3) MSHR merge cap: each cluster pays its own lookup latency before merging
  {
    gpu_l2_tlb t(SETS, ASSOC, MSHR_E, MAXMERGE, LAT, CH, GPU_PAGE_SHIFT);
    uint64_t va = 12345ULL << GPU_PAGE_SHIFT;
    #define L2_DRV(CL,TB) ( (void)t.access(va,(CL),out,(TB)), \
                            t.access(va,(CL),out,(TB)+LAT) )
    assert(t.access(va,0,out,10)==gpu_l2_tlb::LOOKUP_PENDING);
    assert(t.access(va,0,out,10+LAT)==gpu_l2_tlb::MISS && "V-L2: first miss creates MSHR (cluster 0)");
    for (unsigned c=1;c<MAXMERGE;c++)
      assert(L2_DRV(c,200)==gpu_l2_tlb::MSHR_HIT && "V-L2: merge under cap (pays latency first)");
    assert(L2_DRV(MAXMERGE,200)==gpu_l2_tlb::MSHR_FULL && "V-L2: merge cap -> MSHR_FULL");
    #undef L2_DRV
  }

  // (4) MSHR table full + overflow recovery
  {
    gpu_l2_tlb t(SETS, ASSOC, MSHR_E, MAXMERGE, LAT, CH, GPU_PAGE_SHIFT);
    #define L2_RES2(VPN,CL,T) ( (void)t.access((uint64_t)(VPN)<<GPU_PAGE_SHIFT,(CL),out,(T)), \
                                t.access((uint64_t)(VPN)<<GPU_PAGE_SHIFT,(CL),out,(T)+LAT) )
    for (unsigned i=0;i<MSHR_E;i++)             // 128 distinct in-flight VPNs
      assert(L2_RES2(100000+i, 0, 10)==gpu_l2_tlb::MISS && "V-L2: distinct miss fills MSHR table");
    assert(L2_RES2(999999, 0, 10)==gpu_l2_tlb::MSHR_FULL && "V-L2: MSHR table full (graceful)");
    t.remove_from_mshr(100000, 0);              // a rollback / drain frees one slot
    assert(L2_RES2(999999, 0, 5000)==gpu_l2_tlb::MISS && "V-L2: recovers after a slot frees");
    #undef L2_RES2
  }

  printf("[XLAT_VERIFY V-L2] set-assoc/LRU/latency/MSHR/merge/overflow OK : PASS\n");
}

// V-PS-6: 64KB big-page table index decode (pt_pt64k_idx).
// Oracle = the spec bit-field (VA[20:16]), spelled out with literals, never the
// function under test.
static void verify_pt64k_index_decode() {
  printf("[XLAT_VERIFY V-PS-6] --- 64KB PT index (VA[20:16]) ---\n");

  // (1) value = bits [20:16]
  assert(pt_pt64k_idx((uint64_t)5  << 16) == 5  && "V-PS-6: idx 5");
  assert(pt_pt64k_idx((uint64_t)31 << 16) == 31 && "V-PS-6: idx 31 (max)");
  assert(pt_pt64k_idx(0) == 0 && "V-PS-6: idx 0");
  // (2) offset bits [15:0] do not leak into the index
  assert(pt_pt64k_idx(((uint64_t)7 << 16) | 0xFFFF) == 7 && "V-PS-6: low bits ignored");
  // (3) bits at/above 21 do not leak in (field is 5 bits wide)
  assert(pt_pt64k_idx((uint64_t)1 << 21) == 0 && "V-PS-6: bit21 excluded");
  assert(pt_pt64k_idx(((uint64_t)0x3F) << 16) == 0x1F && "V-PS-6: masked to 5 bits");
  // (4) 32 distinct slots tile exactly one 2MB region (independent count)
  for (unsigned k = 0; k < 32; k++)
    assert(pt_pt64k_idx((uint64_t)k << 16) == k && "V-PS-6: slot k");

  printf("[XLAT_VERIFY V-PS-6] 64KB index decode OK : PASS\n");
}

// V-PS-7/8/9: 64KB demand build via the real path (ensure_node x3 + map_leaf),
// exactly as done_tlb_req does per fault. Proves: leaf lives at level 3 in a
// SEPARATE big-PT node (not the PDE, not the 4KB PT), no PS bit, 16 contiguous
// data frames, and true 64KB coverage.
static void verify_64k_demand_build(gpu_module* mod) {
  printf("[XLAT_VERIFY V-PS-7] --- 64KB demand build ---\n");

  const unsigned C  = 0;
  const unsigned P  = 300;   // collision-free pml4 subtree (distinct from V-PT-2's 256)
  chiplet*        chip0 = mod->get_chiplet(C);
  gpu_page_table* pt    = mod->get_page_table();

  // Register the whole 2MB region as 64KB pages so page_shift_for_va() -> 16.
  // This is the real mechanism a 64KB cudaMalloc would use (input, not oracle).
  uint64_t region = build_va(P, 0, 0, 0, 0);
  mod->on_malloc(region, GPU_HUGE_PAGE_SIZE, GPU_64K_SHIFT);

  uint64_t va = region + ((uint64_t)5 << GPU_64K_SHIFT);  // 64k slot 5

  size_t f0 = chip0->free_page_count();
  pt->ensure_node(va, 1, C);   // PDP
  pt->ensure_node(va, 2, C);   // PD
  uint64_t big_pt = pt->ensure_node(va, 3, C);  // 64KB big PT (m_pt64k_nodes)
  pt->map_leaf(va, C);         // 16 contiguous data frames
  size_t f1 = chip0->free_page_count();

  // V-PS-8: conservation -> PDP + PD + big-PT + 16 data = 19 (oracle: 3 nodes +
  // GPU_64K_FRAMES from the spec, independent of the allocator internals).
  assert(f0 - f1 == 3 + GPU_64K_FRAMES && "V-PS-8: 64KB build consumes 3 nodes + 16 frames");

  int d = 0;
  const gpu_pte_t* w = mod->walk_va(va, &d);
  assert(w && w->present() && "V-PS-7: 64KB leaf present");
  assert(d == 4 && "V-PS-7: 64KB walk depth 4 (leaf at level 3, like 4KB)");
  assert(!w->huge_page() && "V-PS-7: 64KB leaf must NOT set the PS bit");

  // Leaf sits in the big-PT node, addressed by the 5-bit index (independent
  // recompute of the PA the GMMU would issue for the level-3 read).
  uint64_t node_pa = pt->get_node_base_pa(va, 3);
  assert(node_pa == big_pt && "V-PS-7: get_node_base_pa(3) == ensure_node(3)");
  assert(node_pa != (uint64_t)-1 && "V-PS-7: big-PT node exists");

  // V-PS-8: data frame is 64KB-aligned (16-frame aligned).
  assert((w->pfn() % GPU_64K_FRAMES) == 0 && "V-PS-8: 64KB data base is 16-frame aligned");

  // V-PS-9: 64KB coverage -- every 4KB offset inside the page shares one
  // translation; the next 64KB slot does not (oracle: definition of a 64KB page).
  for (uint64_t off = 0; off < GPU_64K_PAGE_SIZE; off += GPU_PAGE_SIZE) {
    const gpu_pte_t* wi = mod->walk_va(va + off);
    assert(wi && wi->pfn() == w->pfn() && "V-PS-9: whole 64KB maps to one leaf");
  }
  assert(mod->walk_va(va + GPU_64K_PAGE_SIZE) == nullptr &&
         "V-PS-9: neighbouring 64KB slot is a separate (unmapped) page");

  printf("[XLAT_VERIFY V-PS-7] 64KB build/depth/coverage/PMM OK : PASS\n");
}

// V-PS-10: 4KB PT and 64KB big-PT coexisting under ONE PD entry. This is the
// case K6 never exercises, and the one most likely to silently mistranslate.
static void verify_pagesize_coexistence(gpu_module* mod) {
  printf("[XLAT_VERIFY V-PS-10] --- 4KB + 64KB under one PD entry ---\n");

  const unsigned C = 0;
  const unsigned P = 301;    // fresh subtree
  gpu_page_table* pt = mod->get_page_table();

  uint64_t region = build_va(P, 0, 0, 0, 0);   // one 2MB region == one PD entry
  // Two sub-ranges in the SAME region with different page sizes (inputs).
  uint64_t va64k = region + ((uint64_t)0 << GPU_64K_SHIFT);  // slot 0 -> 64KB
  uint64_t va4k  = region + ((uint64_t)1 << GPU_64K_SHIFT);  // slot 1 region -> 4KB
  mod->on_malloc(va64k, GPU_64K_PAGE_SIZE, GPU_64K_SHIFT);
  mod->on_malloc(va4k,  GPU_64K_PAGE_SIZE, GPU_PAGE_SHIFT);

  // Build the 64KB mapping first.
  pt->ensure_node(va64k, 1, C);
  pt->ensure_node(va64k, 2, C);
  pt->ensure_node(va64k, 3, C);
  pt->map_leaf(va64k, C);
  const gpu_pte_t* w64_before = mod->walk_va(va64k);
  assert(w64_before && w64_before->present() && "V-PS-10: 64KB mapped");
  uint64_t pfn64_before = w64_before->pfn();

  // Now build a 4KB mapping under the SAME PD entry.
  pt->ensure_node(va4k, 1, C);   // reuses PDP
  pt->ensure_node(va4k, 2, C);   // reuses PD
  pt->ensure_node(va4k, 3, C);   // creates the 4KB PT (m_pt_nodes), NOT the big PT
  pt->map_leaf(va4k, C);
  const gpu_pte_t* w4 = mod->walk_va(va4k);
  assert(w4 && w4->present() && !w4->huge_page() && "V-PS-10: 4KB mapped");

  // The two leaf tables are physically distinct nodes hanging off one PD entry.
  uint64_t node64 = pt->get_node_base_pa(va64k, 3);  // -> m_pt64k_nodes
  uint64_t node4  = pt->get_node_base_pa(va4k, 3);   // -> m_pt_nodes
  assert(node64 != (uint64_t)-1 && node4 != (uint64_t)-1 && "V-PS-10: both PT nodes exist");
  assert(node64 != node4 && "V-PS-10: 4KB PT and 64KB big-PT are distinct nodes");

  // Same PD entry: pd_idx identical (independent recompute from the VAs).
  assert(pt_pd_idx(va64k) == pt_pd_idx(va4k) &&
         pt_pdp_idx(va64k) == pt_pdp_idx(va4k) &&
         pt_pml4_idx(va64k) == pt_pml4_idx(va4k) &&
         "V-PS-10: both VAs share one PD entry");

  // Building the 4KB mapping must not corrupt the 64KB translation.
  const gpu_pte_t* w64_after = mod->walk_va(va64k);
  assert(w64_after && w64_after->present() && "V-PS-10: 64KB still mapped");
  assert(w64_after->pfn() == pfn64_before && "V-PS-10: 64KB translation unchanged");
  assert(w64_after->pfn() != w4->pfn() && "V-PS-10: 4KB and 64KB have distinct data");

  printf("[XLAT_VERIFY V-PS-10] coexistence (no cross-corruption) OK : PASS\n");
}

// V-PS-11: three page sizes side by side (fresh subtrees). Proves walk depth and
// PS bit differ exactly as the spec says: 4KB/64KB stop at level 3 (depth 4),
// 2MB stops at the PD (depth 3, PS=1).
static void verify_mixed_pagesizes(gpu_module* mod) {
  printf("[XLAT_VERIFY V-PS-11] --- 4KB / 64KB / 2MB side by side ---\n");

  const unsigned C = 0;
  gpu_page_table* pt = mod->get_page_table();

  struct { unsigned pml4; unsigned shift; int depth; bool huge; } cases[] = {
    { 302, GPU_PAGE_SHIFT, 4, false },
    { 303, GPU_64K_SHIFT,  4, false },
    { 304, GPU_HUGE_SHIFT, 3, true  },
  };
  uint64_t data_pfn[3];
  for (int i = 0; i < 3; i++) {
    uint64_t region = build_va(cases[i].pml4, 0, 0, 0, 0);
    mod->on_malloc(region, GPU_HUGE_PAGE_SIZE, cases[i].shift);
    uint64_t va = region + 0x40;  // within the first leaf page

    pt->ensure_node(va, 1, C);
    pt->ensure_node(va, 2, C);
    if (cases[i].shift != GPU_HUGE_SHIFT)  // 2MB has no level-3 node
      pt->ensure_node(va, 3, C);
    pt->map_leaf(va, C);

    int d = 0;
    const gpu_pte_t* w = mod->walk_va(va, &d);
    assert(w && w->present() && "V-PS-11: mapped");
    assert(d == cases[i].depth && "V-PS-11: walk depth matches page size");
    assert(w->huge_page() == cases[i].huge && "V-PS-11: PS bit matches page size");
    data_pfn[i] = w->pfn();
  }
  // Three separate allocations -> three distinct data pages.
  assert(data_pfn[0] != data_pfn[1] && data_pfn[0] != data_pfn[2] &&
         data_pfn[1] != data_pfn[2] && "V-PS-11: distinct data PFNs");

  printf("[XLAT_VERIFY V-PS-11] mixed page sizes OK : PASS\n");
}

static void run_address_translation_verify(gpu_module* mod) {
  (void)mod;  // V-PT-1 needs no live state; later steps will use mod.
  printf("\n[XLAT_VERIFY] ===== Address Translation Verification =====\n");
  verify_va_index_decode();   // V-PT-1  <- add new step calls here as they grow
  verify_pte_bitfields();     // V-PT-3
  verify_pt_demand_build(mod); // V-PT-2
  verify_pwc(mod);             // V-PWC
  verify_l1_tlb();            // V-L1
  verify_l2_tlb();            // V-L2
  verify_pt64k_index_decode();    // V-PS-6
  verify_64k_demand_build(mod);   // V-PS-7/8/9
  verify_pagesize_coexistence(mod); // V-PS-10
  verify_mixed_pagesizes(mod);    // V-PS-11
  printf("[XLAT_VERIFY] ===== step(s) PASSED =====\n\n");
  fflush(stdout);
}
#endif  // ADDR_XLAT_VERIFY