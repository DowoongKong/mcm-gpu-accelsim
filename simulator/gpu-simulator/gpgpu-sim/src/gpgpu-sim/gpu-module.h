#ifndef GPU_MODULE_H
#define GPU_MODULE_H

#include <assert.h>
#include <stdio.h>
#include <vector>
#include "../lasp/lasp_simple.h"
#include "../abstract_hardware_model.h"

#include <queue>
#include <cstdint>
#include <unordered_map>  // page table node maps
#include <algorithm>   // sort / binary_search in chiplet::finalize_exact_reservations
#include <functional>  // std::greater

// Debug logs: build with -DLOG_TLB, -DLOG_PAGE_TABLE, -DLOG_MEM_ACCESS, -DLOG_PTW or -DLOG_PMM.

#define GPU_PAGE_SHIFT 12  // base (4KB) page
#define GPU_PAGE_SIZE  (1ULL << GPU_PAGE_SHIFT)
// 2MB huge page: the PD (level 2) entry is the leaf, marked by the PS bit (as in Pascal's PDE0).
#define GPU_HUGE_SHIFT     21
#define GPU_HUGE_PAGE_SIZE (1ULL << GPU_HUGE_SHIFT)  // 2MB
#define GPU_HUGE_FRAMES    (GPU_HUGE_PAGE_SIZE / GPU_PAGE_SIZE)  // 512 4KB frames
// 64KB big page: leaf at level 3 in a separate 32-entry table indexed by VA[20:16],
// under the same PD entry as the 4KB PT (Pascal's PDE0 big-page pointer).
#define GPU_64K_SHIFT      16
#define GPU_64K_PAGE_SIZE  (1ULL << GPU_64K_SHIFT)  // 64KB
#define GPU_64K_FRAMES     (GPU_64K_PAGE_SIZE / GPU_PAGE_SIZE)  // 16 4KB frames
// 4-level page table VA index extraction constants (x86-64 style)
#define PT_PML4_SHIFT  39
#define PT_PDP_SHIFT   30
#define PT_PD_SHIFT    21
#define PT_PT_SHIFT    12
#define PT_IDX_MASK    0x1FFULL  // 9-bit index per level (512 entries)

// Forward declaration: prevents circular include (gmmu.h includes gpu-module.h)
class gmmu;
class gpu_l2_tlb;

// 64-bit PTE (x86-64 style): bit 0 P, 1 R/W, 5 A, 6 D, 7 PS (2MB leaf, PD only), 51:12 PFN.
struct gpu_pte_t {
  uint64_t raw;

  // Bit field readers
  bool     present()   const { return  raw        & 1ULL; }
  bool     writable()  const { return (raw >> 1)  & 1ULL; }
  bool     accessed()  const { return (raw >> 5)  & 1ULL; }
  bool     dirty()     const { return (raw >> 6)  & 1ULL; }
  bool     huge_page() const { return (raw >> 7)  & 1ULL; }  // PS bit
  uint64_t pfn()       const { return (raw >> 12) & 0xFFFFFFFFFFULL; }
  uint64_t pa()        const { return pfn() << GPU_PAGE_SHIFT; }

  // Bit field writers
  void set_present (bool v) { v ? (raw |= 1ULL)     : (raw &= ~1ULL); }
  void set_writable(bool v) { v ? (raw |= 1ULL<<1)  : (raw &= ~(1ULL<<1)); }
  void set_accessed(bool v) { v ? (raw |= 1ULL<<5)  : (raw &= ~(1ULL<<5)); }
  void set_dirty   (bool v) { v ? (raw |= 1ULL<<6)  : (raw &= ~(1ULL<<6)); }
  void set_huge_page(bool v){ v ? (raw |= 1ULL<<7)  : (raw &= ~(1ULL<<7)); }  // PS: 2MB leaf
  void set_pfn(uint64_t p) {
    raw = (raw & ~(0xFFFFFFFFFFULL << 12)) | ((p & 0xFFFFFFFFFFULL) << 12);
  }

  bool valid() const { return present(); }
  void clear()       { raw = 0; }
};

// Extract the 9-bit index for each page table level from a virtual address
inline unsigned pt_pml4_idx(uint64_t va) { return (va >> PT_PML4_SHIFT) & PT_IDX_MASK; }
inline unsigned pt_pdp_idx (uint64_t va) { return (va >> PT_PDP_SHIFT)  & PT_IDX_MASK; }
inline unsigned pt_pd_idx  (uint64_t va) { return (va >> PT_PD_SHIFT)   & PT_IDX_MASK; }
inline unsigned pt_pt_idx  (uint64_t va) { return (va >> PT_PT_SHIFT)   & PT_IDX_MASK; }
// 64KB big-page table index: VA[20:16] (32 entries cover the same 2MB as the 4KB PT).
inline unsigned pt_pt64k_idx(uint64_t va) { return (unsigned)((va >> GPU_64K_SHIFT) & 0x1FULL); }
inline uint64_t pt_vpn     (uint64_t va) { return va >> GPU_PAGE_SHIFT; }
inline uint64_t pt_offset  (uint64_t va) { return va & (GPU_PAGE_SIZE - 1); }

// Page-size helpers; shift 12 = 4KB, 16 = 64KB, 21 = 2MB.
inline uint64_t page_vpn    (uint64_t va, unsigned shift) { return va >> shift; }
inline uint64_t page_offset (uint64_t va, unsigned shift) { return va & ((1ULL << shift) - 1); }
inline uint64_t page_base   (uint64_t va, unsigned shift) { return va & ~((1ULL << shift) - 1); }
// PT level that holds the leaf: 2 (PD) for 2MB, 3 for 4KB and 64KB.
inline int      pt_leaf_level(unsigned shift) { return (shift == GPU_HUGE_SHIFT) ? 2 : 3; }
// Log tag for a page size, so 4k/64k/2m stats stay distinguishable.
inline const char* page_kind_tag(unsigned shift) {
  return (shift == GPU_HUGE_SHIFT) ? "2m" : (shift == GPU_64K_SHIFT) ? "64k" : "4k";
}

// One PT node = one 4KB page of 512 PTEs; its PA decides whether a walk step is local or remote.
struct pt_node_t {
  uint64_t  base_pa;       // physical address of this node page (from PMM)
  gpu_pte_t entries[512];  // 512 entries indexed by the 9-bit level index

  // -gmmu_local_ptw: PA of each chiplet's copy of this node (all base_pa without it).
  // Only the address is copied; `entries` stays shared, as walks use the PA only for timing.
  std::vector<uint64_t> base_pa_rep;

  pt_node_t() : base_pa(0) {
    for (int i = 0; i < 512; i++) entries[i].raw = 0;
  }

  uint64_t base_pa_for(unsigned chiplet) const {
    return (chiplet < base_pa_rep.size()) ? base_pa_rep[chiplet] : base_pa;
  }
};

// One registered device allocation: VA range and its page size.
struct va_alloc_t {
  uint64_t va_base;
  size_t   aligned_size;
  // Page shift of this range, fixed at malloc time (read via gpu_module::page_shift_for_va()).
  unsigned page_shift;
};

// GPU-wide 4-level page table. Nodes live in host maps; their base_pa only sets walk timing.
class gpu_page_table {
 public:
  explicit gpu_page_table(class gpu_module* mod);
  ~gpu_page_table();

  // Leaf PTE for va, or nullptr on a fault. walk_depth: levels traversed (verification only).
  const gpu_pte_t* walk(uint64_t va, int* walk_depth = nullptr) const;

  // Create the `level` node (1=PDP, 2=PD, 3=PT) on chiplet_id if absent; returns its base_pa.
  uint64_t ensure_node(uint64_t va, int level, unsigned chiplet_id);
  // Same, with an explicit page_shift (premap runs before any cudaMalloc registers a range).
  uint64_t ensure_node_shift(uint64_t va, int level, unsigned chiplet_id,
                             unsigned page_shift);

  // Allocate the data page for va and set its leaf PTE; the upper nodes must exist.
  void map_leaf(uint64_t va, unsigned chiplet_id);

  // Clear the 4KB leaf PTE for va (not 64KB/2MB); frees nothing.
  void unmap(uint64_t va);

  // PA of the PML4 root (CR3), allocated in the constructor.
  uint64_t get_pml4_base_pa() const {
    assert(m_pml4 != nullptr);
    return m_pml4->base_pa;
  }

  // CR3 of `chiplet`'s page table copy (the shared root without -gmmu_local_ptw).
  uint64_t get_pml4_base_pa(unsigned chiplet) const {
    assert(m_pml4 != nullptr);
    return m_pml4->base_pa_for(chiplet);
  }

  void dump(FILE* f = stdout) const;

  // base_pa of va's node at level 0-3 (PML4/PDP/PD/PT), or (uint64_t)-1 if absent.
  uint64_t get_node_base_pa(uint64_t va, int level) const;

  // Same, for `chiplet`'s copy of the node (local to it under -gmmu_local_ptw).
  uint64_t get_node_base_pa(uint64_t va, int level, unsigned chiplet) const;

  // Premap: set va's leaf PTE to a reserved pfn (explicit page_shift: no cudaMalloc yet).
  void map_leaf_with_pfn(uint64_t va, uint64_t pfn, unsigned page_shift);

  // Allocate the root's per-chiplet copies. Called after premap reserves the data PFNs,
  // so data placement is the same with and without -gmmu_local_ptw.
  void alloc_root_replicas();

  // PT nodes allocated, including the 64KB big PTs (footprint accounting).
  size_t pt_node_count() const {
    return 1 /*PML4*/ + m_pdp_nodes.size() + m_pd_nodes.size()
         + m_pt_nodes.size() + m_pt64k_nodes.size();
  }

  // PT node pages actually consumed from the PMM (x n_chiplets when replicated).
  size_t pt_page_count() const;

  // Placement map I/O. The map is page-size specific; load aborts on a page_shift mismatch.
  bool dump_placement(const char* path) const;
  bool load_placement(const char* path);
  void premap_from_placement();
  // -gmmu_placement_exact 1: premap at the recorded data, node and root PFNs; aborts on mismatch.
  void premap_exact();
  bool placement_has_pfns() const { return m_placement_has_pfns; }
  bool placement_frozen() const { return m_placement_frozen; }

  // First touch: each level's and the data page's owner = chiplet of the first translate
  // (access order, so independent of walk latency). Returns true if anything was inserted.
  bool     record_first_touch(uint64_t va, unsigned chiplet);
  // Deferred first touch: inserts from the parallel passes are buffered and committed after
  // they join, in chiplet order, so the maps match a 1-thread run for any thread count.
  void     note_first_touch(uint64_t va, unsigned chiplet);   // routes below
  void     enable_deferred_first_touch(unsigned n_chiplets);
  bool     deferred_first_touch_enabled() const { return m_defer_ft; }
  void     commit_deferred_first_touch();
  unsigned long long deferred_ft_commits() const { return m_ft_commits; }
  unsigned long long deferred_ft_dedups() const { return m_ft_dedups; }
  unsigned long long deferred_ft_max_batch() const { return m_ft_max_batch; }
  unsigned first_touch_node_owner(uint64_t va, int level) const;  // level 1=PDP,2=PD,3=PT
  unsigned first_touch_data_owner(uint64_t va) const;

 private:
  class gpu_module* m_mod;

  pt_node_t* m_pml4;  // root node; pre-allocated in constructor (always valid)

  // Nodes by key: PDP pml4, PD pml4<<9|pdp, PT pml4<<18|pdp<<9|pd.
  std::unordered_map<uint64_t, pt_node_t*> m_pdp_nodes;
  std::unordered_map<uint64_t, pt_node_t*> m_pd_nodes;
  std::unordered_map<uint64_t, pt_node_t*> m_pt_nodes;
  // 64KB big PTs, keyed like m_pt_nodes (both may hang off one PD entry); 32 entries used.
  std::unordered_map<uint64_t, pt_node_t*> m_pt64k_nodes;

  // First-touch owner maps (first-write-wins). Keys mirror the node keys.
  std::unordered_map<uint64_t, unsigned> m_pdp_owner;   // key: pml4_idx
  std::unordered_map<uint64_t, unsigned> m_pd_owner;    // key: pml4<<9 | pdp
  std::unordered_map<uint64_t, unsigned> m_pt_owner;    // key: pml4<<18 | pdp<<9 | pd
  std::unordered_map<uint64_t, unsigned> m_data_owner;  // key: vpn
  // PFNs read from a v3 placement map (load_placement); used only by premap_exact().
  std::unordered_map<uint64_t, uint64_t> m_file_data_pfn;  // key: vpn
  std::unordered_map<uint64_t, uint64_t> m_file_pdp_pfn;   // key: pml4_idx
  std::unordered_map<uint64_t, uint64_t> m_file_pd_pfn;    // key: pml4<<9 | pdp
  std::unordered_map<uint64_t, uint64_t> m_file_pt_pfn;    // key: pml4<<18 | pdp<<9 | pd
  uint64_t m_file_root_pfn = (uint64_t)-1;
  bool     m_placement_has_pfns = false;   // every N line carried a PFN and an R line was read
  pt_node_t* make_node_at(uint64_t pfn);   // node at a given frame, no PMM call

  // New node from chiplet_id's PMM (nullptr if exhausted); the root passes with_replicas=false.
  pt_node_t* alloc_node(unsigned chiplet_id, bool with_replicas = true);
  bool       alloc_replicas(pt_node_t* node, unsigned owner);

  // Page table change count; gpgpu_sim::cycle() aborts if it moves in the parallel passes.
  unsigned long long m_pt_mutations = 0;
  void note_mutation() { m_pt_mutations++; }

 public:
  unsigned long long mutation_count() const { return m_pt_mutations; }

 private:
  // Deferred first touch. m_ft_buf[c] is written only by chiplet c's GMMU, so no lock is needed.
  bool m_defer_ft = false;
  std::vector<std::vector<uint64_t>> m_ft_buf;
  unsigned long long m_ft_commits = 0;    // buffered entries committed
  unsigned long long m_ft_dedups = 0;     // commits that inserted nothing
  unsigned long long m_ft_max_batch = 0;  // largest single-cycle batch

  // Set by load_placement(): owners come from the file, not from first touch.
  bool m_placement_frozen = false;
  // Page shift declared by the loaded placement map. -1 until load_placement().
  int m_placement_page_shift = -1;
  unsigned config_uniform_page_shift() const;

};

// Forward declarations only (keep dependencies minimal)
class gpgpu_sim;
class simt_core_cluster;
class memory_partition_unit;
class memory_config;  // full definition in gpu-sim.h; pointer-only use here

// A chiplet views a contiguous range of the global clusters and memory partitions (not owned).
class chiplet {
 public:
  chiplet(gpgpu_sim* top, unsigned chiplet_id);
  ~chiplet();

  unsigned id() const { return m_id; }

  // Bind contiguous range for clusters (global IDs)
  void bind_cluster_range(unsigned global_begin, unsigned count);

  // Bind contiguous range for memory partitions (global IDs) and their sub-partitions.
  void bind_mem_range(unsigned global_begin, unsigned count,
                      unsigned sub_per_mem);

  // After gpgpu_sim allocates global arrays, chiplet stores the pointers here.
  void bind_global_arrays(simt_core_cluster** global_clusters,
                          memory_partition_unit** global_mem);

  // Cluster range getters
  unsigned get_cluster_global_begin() const { return m_cluster_global_begin; }
  unsigned get_num_clusters() const { return m_n_clusters_per_chiplet; }

  // Memory partition range getters
  unsigned get_mem_global_begin() const { return m_mem_global_begin; }
  unsigned get_num_mem() const { return m_n_mem_per_chiplet; }
  unsigned get_mem_sub_partition_global_begin() const { return m_mem_sub_partition_global_begin; }
  unsigned get_num_mem_sub_partition() const { return m_n_mem_sub_partition_per_chiplet; }

  // Accessors (global index in the whole GPU)
  simt_core_cluster* get_cluster_global(unsigned global_idx) const;
  memory_partition_unit* get_mem_global(unsigned global_idx) const;

  // Accessors (local index within this chiplet)
  simt_core_cluster* get_cluster(unsigned local_idx) const;
  memory_partition_unit* get_mem(unsigned local_idx) const;
  
  // PMM: bump allocator over [m_pfn_begin, m_pfn_end), with aligned runs for 64KB/2MB pages.
  // m_recycled holds frames skipped for alignment or left free by an exact replay.
  void pmm_set_range(uint64_t pfn_begin, uint64_t pfn_end) {
    m_pfn_begin = pfn_begin; m_pfn_end = pfn_end; m_pfn_next = pfn_begin;
  }
  // One 4KB frame. Reuses alignment leftovers first so they are not wasted.
  uint64_t alloc_page() {
    if (!m_recycled.empty()) {
      uint64_t pfn = m_recycled.back(); m_recycled.pop_back();
      return pfn;
    }
    if (m_pfn_next >= m_pfn_end) return (uint64_t)-1;
    return m_pfn_next++;
  }
  // n_frames contiguous frames aligned to n_frames; skipped frames are recycled.
  uint64_t alloc_contiguous(uint64_t n_frames) {
    uint64_t aligned = (m_pfn_next + n_frames - 1) / n_frames * n_frames;
    if (aligned + n_frames > m_pfn_end) return (uint64_t)-1;
    for (uint64_t pfn = m_pfn_next; pfn < aligned; pfn++) m_recycled.push_back(pfn);
    m_pfn_next = aligned + n_frames;
    return aligned;
  }
  void free_page(uint64_t pfn) { m_recycled.push_back(pfn); }
  // -gmmu_placement_exact: record frames [pfn, pfn + n); false if outside this chiplet's range.
  bool reserve_exact(uint64_t pfn, uint64_t n) {
    if (pfn < m_pfn_begin || pfn >= m_pfn_end || n > m_pfn_end - pfn) return false;
    for (uint64_t k = 0; k < n; k++) m_exact_frames.push_back(pfn + k);
    return true;
  }
  // Take the recorded frames: move the cursor past the highest and recycle free frames below it.
  // Returns false on a duplicate or an already allocated frame.
  bool finalize_exact_reservations() {
    if (m_exact_frames.empty()) return true;
    std::sort(m_exact_frames.begin(), m_exact_frames.end());
    for (size_t i = 1; i < m_exact_frames.size(); i++)
      if (m_exact_frames[i] == m_exact_frames[i - 1]) return false;
    std::vector<uint64_t> recycled_sorted(m_recycled.begin(), m_recycled.end());
    std::sort(recycled_sorted.begin(), recycled_sorted.end());
    for (uint64_t f : m_exact_frames)
      if (f < m_pfn_next && !std::binary_search(recycled_sorted.begin(), recycled_sorted.end(), f))
        return false;   // already handed out
    std::vector<uint64_t> keep;   // old recycled frames that are not replayed
    for (uint64_t f : recycled_sorted)
      if (!std::binary_search(m_exact_frames.begin(), m_exact_frames.end(), f)) keep.push_back(f);
    const uint64_t top = m_exact_frames.back() + 1;
    for (uint64_t f = m_pfn_next; f < top; f++)   // holes between cursor and the top frame
      if (!std::binary_search(m_exact_frames.begin(), m_exact_frames.end(), f)) keep.push_back(f);
    if (top > m_pfn_next) m_pfn_next = top;
    // Descending, so alloc_page() (pops the back) reuses frames in ascending order.
    std::sort(keep.begin(), keep.end(), std::greater<uint64_t>());
    m_recycled.swap(keep);
    m_exact_count = m_exact_frames.size();
    m_exact_frames.clear();
    return true;
  }
  size_t exact_frames_taken() const { return m_exact_count; }
  uint64_t pfn_begin() const { return m_pfn_begin; }
  uint64_t pfn_end() const { return m_pfn_end; }
  size_t free_page_count() const {
    return (size_t)(m_pfn_end - m_pfn_next) + m_recycled.size();
  }

  // Per-chiplet L2 TLB accessors, one TLB per page size.
  gpu_l2_tlb* get_l2_tlb_4k()  const { return m_l2_tlb_4k; }
  gpu_l2_tlb* get_l2_tlb_64k() const { return m_l2_tlb_64k; }
  gpu_l2_tlb* get_l2_tlb_2m()  const { return m_l2_tlb_2m; }
  // Picks the TLB for a page shift (12 -> 4k, 16 -> 64k, 21 -> 2m).
  gpu_l2_tlb* get_l2_tlb(unsigned page_shift) const {
    if (page_shift == GPU_HUGE_SHIFT) return m_l2_tlb_2m;
    if (page_shift == GPU_64K_SHIFT)  return m_l2_tlb_64k;
    return m_l2_tlb_4k;
  }
  void set_l2_tlb_4k (gpu_l2_tlb* t) { m_l2_tlb_4k  = t; }
  void set_l2_tlb_64k(gpu_l2_tlb* t) { m_l2_tlb_64k = t; }
  void set_l2_tlb_2m (gpu_l2_tlb* t) { m_l2_tlb_2m  = t; }

 private:
  gpgpu_sim* m_top;
  unsigned m_id;

  // Range binding (global IDs)
  unsigned m_cluster_global_begin;
  unsigned m_n_clusters_per_chiplet;

  unsigned m_mem_global_begin;
  unsigned m_n_mem_per_chiplet;

  unsigned m_mem_sub_partition_global_begin;
  unsigned m_n_mem_sub_partition_per_chiplet;

  // Pointers to global arrays (NOT owned)
  simt_core_cluster** m_global_clusters;
  memory_partition_unit** m_global_mem;

  // PMM state: bump cursor and recycled frames.
  uint64_t m_pfn_begin = 0, m_pfn_end = 0, m_pfn_next = 0;
  std::vector<uint64_t> m_recycled;
  std::vector<uint64_t> m_exact_frames;   // frames recorded by reserve_exact()
  size_t m_exact_count = 0;               // frames taken by the last finalize

  // Per-chiplet L2 TLBs, created in gpu_module::init(). One per page size.
  gpu_l2_tlb* m_l2_tlb_4k  = nullptr;
  gpu_l2_tlb* m_l2_tlb_64k = nullptr;
  gpu_l2_tlb* m_l2_tlb_2m  = nullptr;
};

// The whole MCM GPU: chiplets, device -> chiplet owner maps, page table, GMMUs, L2 TLBs.
class gpu_module {
 public:
  gpu_module(gpgpu_sim* top, unsigned n_chiplets, unsigned n_clusters,
             unsigned n_mem, unsigned n_mem_sub_partition);
  ~gpu_module();

  // Contiguous split per chiplet (split_range(): the remainder goes to the lowest indices).
  void init();
  void create_chiplets();
  static void split_range(unsigned total, unsigned n_mem, unsigned part_id,
                          unsigned* out_begin, unsigned* out_count);
  // Bind the global cluster and memory partition arrays, once gpgpu_sim has allocated them.
  void bind_global_arrays(simt_core_cluster** global_clusters,
                          memory_partition_unit** global_mem);

  // Hooking push and pop
  void icnt_push(unsigned input_deviceID, unsigned output_deviceID,
                 void* data, unsigned int size);
  void* icnt_pop(unsigned output_deviceID);
  
  // Set stats file for logging MCM access
  void set_stats_file(FILE* fout) { m_stats_file = fout; }
  
  // MCM access totals (fallback counter + all thread slots); read only when printing stats.
  unsigned long long get_remote_access_count() const { return acc_total(0); }
  unsigned long long get_local_access_count() const { return acc_total(1); }

  // Getters
  chiplet* get_chiplet(unsigned i) const { return m_chiplets[i]; }
  unsigned get_num_chiplets() const { return m_n_chiplets; }
  // Which chiplet owns global memory partition `mp` (used by the deadlock dump).
  unsigned chiplet_of_mem(unsigned mp) const {
    return (mp < m_mem_to_chiplet.size()) ? m_mem_to_chiplet[mp] : 0;
  }

  // -gmmu_local_ptw: each chiplet walks its own copy of the PT nodes; data pages are not copied.
  bool local_ptw_enabled() const;

  // Page shift every allocation gets under this config, or 0 if it depends on size (policies 3-5).
  unsigned uniform_page_shift() const;
  unsigned get_num_clusters() const { return m_n_clusters; }
  unsigned get_num_mem() const { return m_n_mem; }
  unsigned get_num_mem_sub_partition() const { return m_n_mem_sub_partition; }
  // Provides access to memory config for GMMU child_mf creation.
  const memory_config* get_memory_config() const;

  // PMM interface: routes page allocation to the target chiplet.
  uint64_t pmm_alloc_page(unsigned chiplet_id);
  // n_frames contiguous, n_frames-aligned 4KB frames (64KB/2MB data pages); -1 if none.
  uint64_t pmm_alloc_contiguous(unsigned chiplet_id, uint64_t n_frames);
  // Exact-PFN replay (see chiplet::reserve_exact / finalize_exact_reservations).
  bool     pmm_reserve_exact(unsigned chiplet_id, uint64_t pfn, uint64_t n_frames);
  bool     pmm_finalize_exact(unsigned chiplet_id);
  size_t   pmm_exact_frames_taken(unsigned chiplet_id) const;

  // Walk the page table for va. Returns the leaf PTE, or nullptr on page fault.
  const gpu_pte_t* walk_va(uint64_t va, int* walk_depth = nullptr) const;
  
  // Register [va_base, va_base + aligned_size): at cudaMalloc, or at each MemcpyHtoD in traces.
  void on_malloc(uint64_t va_base, size_t aligned_size, unsigned page_shift);
  // Returns true if va falls within any registered cudaMalloc range.
  bool is_va_allocated(uint64_t va) const;
  // Page shift for va: its range's, else the fixed policy's size, else 4KB.
  unsigned page_shift_for_va(uint64_t va) const;
  // Returns the GMMU for chiplet_id. Valid after init().
  class gmmu* get_gmmu(unsigned chiplet_id) const;

  // L1 TLB miss -> L2 TLB, then GMMU on a miss. out_pte is set on an L2 hit (is_l2_hit).
  // Returns true if accepted (hit, new walk, merge); false means retry later.
  bool l2_tlb_request(uint64_t va, unsigned cluster_id, uint64_t cycle,
                      gpu_pte_t& out_pte, bool& is_l2_hit,
                       bool* link_used = nullptr,
                       uint64_t* pending_until = nullptr,   // LOOKUP_PENDING: cycle the lookup ends
                       unsigned* full_kind = nullptr);      // MSHR_FULL: 1=merge cap, 2=table full


  // Direct access to the page table
  gpu_page_table* get_page_table() const { return m_page_table; }

  void     pmm_free_page(uint64_t pfn, unsigned chiplet_id);
  unsigned pfn_to_chiplet_via_addrdec(uint64_t pa) const;

  bool is_remote_access(unsigned inputDeviceID,
                            unsigned outputDeviceID,
                            void* data) const;

  unsigned cluster_to_chiplet(unsigned global_cluster_id) const;
  unsigned mem_sub_partition_to_chiplet(unsigned global_mem_part_id) const;

  // Device -> chiplet owner maps, for interconnect setup.
  const std::vector<unsigned>& get_cluster_to_chiplet_vec() const {
    return m_cluster_to_chiplet;
  }
  const std::vector<unsigned>& get_mem_sub_partition_to_chiplet_vec() const {
    return m_mem_sub_partition_to_chiplet;
  }

  unsigned cluster2device(unsigned clusterid) const { return clusterid; }
  unsigned device2cluster(unsigned deviceid) const { return deviceid; }
  unsigned mem2device(unsigned memid) const { return memid + m_n_clusters; }
  unsigned device2mem(unsigned deviceid) const { return deviceid - m_n_clusters; }

  // Debug: print partitioning results
  void dump_partitioning(FILE* fout = stdout) const;
  void dump_mapping_ranges(FILE* fout, const char* tag,
                         const std::vector<unsigned>& chiplet_mapper) const;

  // Debug: print owner maps; L2 slices too if sub_partition_per_mem > 0.
  void dump_mapping(unsigned sub_partition_per_mem,
                       bool verbose_per_id = false,
                       FILE* fout = stdout) const;

 private:
  gpgpu_sim* m_top;
  unsigned m_n_chiplets;
  unsigned m_n_clusters;
  unsigned m_n_mem;
  unsigned m_n_mem_sub_partition;

  // chiplet objects
  std::vector<chiplet*> m_chiplets;

  // Owner maps: global cluster / memory partition / sub-partition id -> chiplet id.
  std::vector<unsigned> m_cluster_to_chiplet;
  std::vector<unsigned> m_mem_to_chiplet;
  std::vector<unsigned> m_mem_sub_partition_to_chiplet;

  // global array pointers (NOT owned)
  simt_core_cluster** m_global_clusters;
  memory_partition_unit** m_global_mem;
  
  // Stats file and counters for MCM access logging
  FILE* m_stats_file;
  // Shared fallback counters, used only when no per-thread slot is usable.
  unsigned long long m_remote_access_count;
  unsigned long long m_local_access_count;

  // Per-thread MCM access counter slots: icnt_deliver() also runs in the parallel phase.
  // Stride 8 = one 64 B cache line per slot; increments stay atomic, so counts are exact.
  enum { ACC_SLOT_STRIDE = 8, ACC_KINDS = 2 };  // [slot*8 + kind], kind 0=remote 1=local
  unsigned m_acc_slots;                      // number of slots (0 = fallback only)
  std::vector<unsigned long long> m_acc_counts;  // size = m_acc_slots * ACC_SLOT_STRIDE

  // Sum for statistics. Never called inside a parallel region (print time only).
  unsigned long long acc_total(unsigned kind) const {
    unsigned long long t = (kind == 0) ? m_remote_access_count : m_local_access_count;
    for (unsigned s = 0; s < m_acc_slots; ++s)
      t += m_acc_counts[(size_t)s * ACC_SLOT_STRIDE + kind];
    return t;
  }

  // kind 0 = remote, 1 = local. Out of line: the body depends on _OPENMP (ODR across TUs).
  void bump_access_count(unsigned kind);
  
  gpu_page_table* m_page_table;  // single GPU-wide page table
  
  std::vector<va_alloc_t> m_va_alloc_list;  // registered VA ranges (see on_malloc)
  // uniform_page_shift() cached at init; nonzero lets page_shift_for_va() skip the range scan.
  unsigned m_uniform_shift = 0;
  std::vector<class gmmu*> m_gmmus;          // one GMMU per chiplet, indexed by chiplet_id
  
  // Push to the ICNT and count local/remote; inter-chiplet latency is modeled by the ring.
  void icnt_deliver(unsigned in, unsigned out, void* data, unsigned size);

};

#endif  // GPU_MODULE_H
