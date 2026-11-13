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
#include <algorithm>   // s30: sort / binary_search in chiplet::finalize_exact_reservations
#include <functional>  // s30: std::greater

// #define LOG_TLB
// #define LOG_PAGE_TABLE
// #define LOG_MEM_ACCESS
// #define LOG_PTW
// #define LOG_PMM

#define GPU_PAGE_SHIFT 12  // base (4KB) page
#define GPU_PAGE_SIZE  (1ULL << GPU_PAGE_SHIFT)
// huge page (2MB): the PD (level 2) entry itself is the leaf, flagged by the PS
// bit. Mirrors Pascal's PDE0 "Valid=1 means this entry is a 2MB PTE".
#define GPU_HUGE_SHIFT     21
#define GPU_HUGE_PAGE_SIZE (1ULL << GPU_HUGE_SHIFT)  // 2MB
#define GPU_HUGE_FRAMES    (GPU_HUGE_PAGE_SIZE / GPU_PAGE_SIZE)  // 512 4KB frames
// big page (64KB): leaf still lives at level 3, but in a separate 32-entry page
// table indexed by VA[20:16], hanging off the same PD entry as the 4KB PT.
// Mirrors Pascal's PDE0 big-page pointer (AB/PAB).
#define GPU_64K_SHIFT      16
#define GPU_64K_PAGE_SIZE  (1ULL << GPU_64K_SHIFT)  // 64KB
#define GPU_64K_FRAMES     (GPU_64K_PAGE_SIZE / GPU_PAGE_SIZE)  // 16 4KB frames
#define GPU_64K_PT_ENTRIES 32  // 2MB coverage / 64KB per entry
// new: 4-level page table VA index extraction constants (x86-64 style)
#define PT_PML4_SHIFT  39
#define PT_PDP_SHIFT   30
#define PT_PD_SHIFT    21
#define PT_PT_SHIFT    12
#define PT_IDX_MASK    0x1FFULL  // 9-bit index per level (512 entries)

// Forward declaration: prevents circular include (gmmu.h includes gpu-module.h) (P3)
class gmmu;
class gpu_l2_tlb;

// new : 64-bit PTE format (x86-64 style)
// Bits 63-52: reserved (0)
// Bits 51-12: PFN (40 bits, physical frame number)
// Bit  7    : PS  (Page Size; 1 = 2MB huge page, used in PD level only)
// Bit  6    : D   (Dirty)
// Bit  5    : A   (Accessed)
// Bit  1    : R/W (Writable)
// Bit  0    : P   (Present)
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

// new: extract 9-bit index for each page table level from a virtual address
inline unsigned pt_pml4_idx(uint64_t va) { return (va >> PT_PML4_SHIFT) & PT_IDX_MASK; }
inline unsigned pt_pdp_idx (uint64_t va) { return (va >> PT_PDP_SHIFT)  & PT_IDX_MASK; }
inline unsigned pt_pd_idx  (uint64_t va) { return (va >> PT_PD_SHIFT)   & PT_IDX_MASK; }
inline unsigned pt_pt_idx  (uint64_t va) { return (va >> PT_PT_SHIFT)   & PT_IDX_MASK; }
// 64KB big-page table index: VA[20:16], 5 bits, 32 entries covering the same 2MB
// as the 512-entry 4KB PT.
inline unsigned pt_pt64k_idx(uint64_t va) { return (unsigned)((va >> GPU_64K_SHIFT) & 0x1FULL); }
inline uint64_t pt_vpn     (uint64_t va) { return va >> GPU_PAGE_SHIFT; }
inline uint64_t pt_offset  (uint64_t va) { return va & (GPU_PAGE_SIZE - 1); }

// Page-size aware helpers. The shift alone identifies the page kind:
// GPU_PAGE_SHIFT (12) = 4KB, GPU_64K_SHIFT (16) = 64KB, GPU_HUGE_SHIFT (21) = 2MB.
inline uint64_t page_vpn    (uint64_t va, unsigned shift) { return va >> shift; }
inline uint64_t page_offset (uint64_t va, unsigned shift) { return va & ((1ULL << shift) - 1); }
inline uint64_t page_base   (uint64_t va, unsigned shift) { return va & ~((1ULL << shift) - 1); }
// Which PT level holds the leaf entry for this page size: 2MB stops at the PD
// (level 2); 4KB and 64KB both walk down to level 3 and differ only in which node
// lives there (m_pt_nodes vs m_pt64k_nodes). Single source of truth for the walk
// termination condition.
inline int      pt_leaf_level(unsigned shift) { return (shift == GPU_HUGE_SHIFT) ? 2 : 3; }
// Log tag for a page size, so 4k/64k/2m stats stay distinguishable.
inline const char* page_kind_tag(unsigned shift) {
  return (shift == GPU_HUGE_SHIFT) ? "2m" : (shift == GPU_64K_SHIFT) ? "64k" : "4k";
}

// new: one page table node = one 4KB physical page holding 512 PTEs
// base_pa is used by Phase 6 PTW to determine local vs remote DRAM access latency
struct pt_node_t {
  uint64_t  base_pa;       // physical address of this node page (from PMM)
  gpu_pte_t entries[512];  // 512 entries indexed by the 9-bit level index

  // -gmmu_local_ptw: per-chiplet copies of THIS node's page, so a walk from
  // chiplet c reads a copy that lives in c's own PA range. Baseline fills every
  // slot with base_pa, i.e. the single shared page -- identical to the behaviour
  // before this option existed.
  //
  // Only the ADDRESS is per-chiplet; `entries` stays shared. A walk in this
  // simulator resolves the next level through the C++ node maps and issues a
  // DRAM read to the node PA purely for timing -- it never reads PTE bytes out
  // of simulated memory. Under frozen placement the page table is read-only, so
  // the copies cannot diverge.
  // (4_/docs/local_ptw_experiment_design.ko.md 1.5, 2.2)
  std::vector<uint64_t> base_pa_rep;

  pt_node_t() : base_pa(0) {
    for (int i = 0; i < 512; i++) entries[i].raw = 0;
  }

  uint64_t base_pa_for(unsigned chiplet) const {
    return (chiplet < base_pa_rep.size()) ? base_pa_rep[chiplet] : base_pa;
  }
};

// Tracks one cudaMalloc VA allocation for GMMU fault validation. (P3)
struct va_alloc_t {
  uint64_t va_base;
  size_t   aligned_size;
  // Page size of this VA range, fixed at malloc time (12 = 4KB, 21 = 2MB).
  // Everything downstream (TLB routing, PT walk depth, PA rebuild) derives from
  // this via gpu_module::page_shift_for_va().
  unsigned page_shift;
};

// new: software-model 4-level radix tree page table (one per GPU)
// Node data lives in host memory (unordered_map); only base_pa reflects
// the GPU DRAM location for PTW latency simulation in Phase 6.
class gpu_page_table {
 public:
  explicit gpu_page_table(class gpu_module* mod);
  ~gpu_page_table();

  // Walk 4 levels for va. Returns pointer to the leaf PTE, or nullptr if
  // any level is not present (signals a page fault to the caller).
  // walk_depth: set to the number of PT levels actually traversed (0-4);
  // used by Phase 6 GMMU to compute per-step DRAM access latency.
  const gpu_pte_t* walk(uint64_t va, int* walk_depth = nullptr) const;

  // Ensure the PT structural node at `level` for va exists.
  // Creates only the node at this level and updates the parent PTE to P=1.
  // Does NOT create nodes at deeper levels or allocate the data page.
  // level: 1=PDP, 2=PD, 3=PT  (level 0 = PML4 is always pre-allocated in constructor)
  // chiplet_id: PMM source for the new node (first-touch placement).
  // Returns base_pa of the node at this level (newly created or already existing).
  uint64_t ensure_node(uint64_t va, int level, unsigned chiplet_id); // (P3-2)
  // Same, but the level-3 map (m_pt_nodes vs m_pt64k_nodes) is chosen from an
  // explicit page_shift instead of page_shift_for_va(). Needed by premap, which
  // runs before any cudaMalloc has registered a VA range.
  uint64_t ensure_node_shift(uint64_t va, int level, unsigned chiplet_id,
                             unsigned page_shift);

  // Allocate the data page for va and set the leaf PTE (level-3 fault handler).
  // All intermediate PT nodes (PDP, PD, PT) must already exist before calling this.
  void map_leaf(uint64_t va, unsigned chiplet_id); // (P3-2)

  // Clear the leaf PTE for va (does not free the physical page or PT nodes).
  void unmap(uint64_t va);

  // Returns the physical address of the PML4 root (CR3 equivalent).
  // Always valid: PML4 is pre-allocated in the constructor.
  uint64_t get_pml4_base_pa() const {
    assert(m_pml4 != nullptr);
    return m_pml4->base_pa;
  }

  // Root PA as seen by `chiplet` -- the CR3 of that chiplet's page table copy.
  // Without replication this is the one global root, identical for all.
  uint64_t get_pml4_base_pa(unsigned chiplet) const {
    assert(m_pml4 != nullptr);
    return m_pml4->base_pa_for(chiplet);
  }

  void dump(FILE* f = stdout) const;

  // Returns the base physical address of the PT node at the given walk level for va.
  // level: 0=PML4, 1=PDP, 2=PD, 3=PT.  Returns (uint64_t)-1 if the node is absent.
  // Phase 6 PTW uses this to compute per-level DRAM access addresses and determine
  // local vs remote latency for each step of the page table walk.
  uint64_t get_node_base_pa(uint64_t va, int level) const;

  // Same, but returns the PA of `chiplet`'s copy of that node. With
  // -gmmu_local_ptw this always lies in `chiplet`'s own PA range, so the walk's
  // DRAM read is local. Without it, identical to the 2-arg form.
  // Applies the same page-size dispatch as the 2-arg form: level 3 reads
  // m_pt64k_nodes for 64KB and m_pt_nodes for 4KB; 2MB has no level 3.
  uint64_t get_node_base_pa(uint64_t va, int level, unsigned chiplet) const;

  // Install a leaf mapping for `va` using an EXACT pfn instead of asking the
  // PMM for the next free one. Needed to replay a placement map bit-for-bit.
  // Picks the leaf level and node map by page size, like map().
  // Install a leaf PTE for an ALREADY-reserved pfn (premap path). page_shift is
  // passed explicitly rather than looked up: during premap no cudaMalloc has run
  // yet, so page_shift_for_va() would answer 4KB for every VA.
  void map_leaf_with_pfn(uint64_t va, uint64_t pfn, unsigned page_shift);

  // Allocate the PML4 root's per-chiplet copies. Deliberately NOT done in the
  // constructor: with -gmmu_local_ptw that would consume one page from every
  // chiplet pool before premap_from_placement() reserves the data-page PFNs,
  // shifting every data PFN by one. (local_ptw_experiment_design.ko.md 2.4)
  void alloc_root_replicas();

  // PT node pages currently allocated (footprint accounting).
  // m_pt64k_nodes must be counted too: with -gpgpu_page_size_policy 1 every
  // level-3 node lives there and omitting it would report a footprint of 0
  // level-3 nodes.
  size_t pt_node_count() const {
    return 1 /*PML4*/ + m_pdp_nodes.size() + m_pd_nodes.size()
         + m_pt_nodes.size() + m_pt64k_nodes.size();
  }

  // PT node pages actually consumed from the PMM (x n_chiplets when replicated).
  size_t pt_page_count() const;

  // Placement snapshot I/O. The map is page-size specific (vpn = va >> shift),
  // so load refuses a file whose page_shift header does not match this config.
  bool dump_placement(const char* path) const;
  bool load_placement(const char* path);
  void premap_from_placement();
  // s30 (-gmmu_placement_exact 1): like premap_from_placement() but every data
  // page, PT node and the root sit at the PFN recorded in a v3 map. Aborts on any
  // inconsistency instead of falling back to re-derived PFNs.
  void premap_exact();
  bool placement_has_pfns() const { return m_placement_has_pfns; }
  bool placement_frozen() const { return m_placement_frozen; }

  // First-touch placement: record the owning chiplet of a VA at its FIRST
  // translate (first L2 miss). Placement then follows access order, not walk
  // completion order, so it is deterministic and PTW-latency-independent.
  // Returns true when at least one owner map actually inserted (i.e. some
  // level of this VA had never been touched before).
  bool     record_first_touch(uint64_t va, unsigned chiplet);
  // A1 (deferred first touch): the only page-table mutation reachable from the
  // parallel CORE phase is the owner-map insert in record_first_touch()
  // (gmmu::translate(), pass B). Every reader of the owner maps runs in the
  // sequential L2/GMMU phase at least one cycle later (walks start the cycle
  // after their translate), so the insert can be buffered during the parallel
  // phases and committed once they have joined -- in canonical order: chiplet
  // index ascending, arrival order within a chiplet. That is exactly the order
  // the 1-thread pass-B loop performs the inserts, so the resulting owner maps
  // are bit-identical to a sequential run's, for any thread count.
  // Gated by RUN MODE (reference = dump && !load), never by thread count: a
  // map must never be a function of OMP_NUM_THREADS.
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

  // Sparse intermediate nodes, keyed by concatenated level indices:
  //   PDP key : pml4_idx               (9 bits)
  //   PD  key : pml4_idx<<9 | pdp_idx  (18 bits)
  //   PT  key : pml4_idx<<18 | pdp_idx<<9 | pd_idx (27 bits)
  std::unordered_map<uint64_t, pt_node_t*> m_pdp_nodes;
  std::unordered_map<uint64_t, pt_node_t*> m_pd_nodes;
  std::unordered_map<uint64_t, pt_node_t*> m_pt_nodes;
  // 64KB big-page tables. Same key as m_pt_nodes (the PD-level prefix): both can
  // hang off one PD entry, which is how Pascal's PDE0 small/big pointer pair is
  // represented here. Only the first GPU_64K_PT_ENTRIES entries are used.
  std::unordered_map<uint64_t, pt_node_t*> m_pt64k_nodes;

  // First-touch owner maps (first-write-wins). Keys mirror the node keys.
  std::unordered_map<uint64_t, unsigned> m_pdp_owner;   // key: pml4_idx
  std::unordered_map<uint64_t, unsigned> m_pd_owner;    // key: pml4<<9 | pdp
  std::unordered_map<uint64_t, unsigned> m_pt_owner;    // key: pml4<<18 | pdp<<9 | pd
  std::unordered_map<uint64_t, unsigned> m_data_owner;  // key: vpn
  // s30: PFNs read from a v3 placement map (load_placement); used only by premap_exact().
  std::unordered_map<uint64_t, uint64_t> m_file_data_pfn;  // key: vpn
  std::unordered_map<uint64_t, uint64_t> m_file_pdp_pfn;   // key: pml4_idx
  std::unordered_map<uint64_t, uint64_t> m_file_pd_pfn;    // key: pml4<<9 | pdp
  std::unordered_map<uint64_t, uint64_t> m_file_pt_pfn;    // key: pml4<<18 | pdp<<9 | pd
  uint64_t m_file_root_pfn = (uint64_t)-1;
  bool     m_placement_has_pfns = false;   // every N line carried a PFN and an R line was read
  pt_node_t* make_node_at(uint64_t pfn);   // s30: node at a given frame, no PMM call

  // Allocate a new PT node page from chiplet_id's PMM.
  // Returns nullptr if PMM is exhausted.
  // with_replicas=false defers the per-chiplet copies (used for the PML4 root,
  // see alloc_root_replicas).
  pt_node_t* alloc_node(unsigned chiplet_id, bool with_replicas = true);
  bool       alloc_replicas(pt_node_t* node, unsigned owner);

  // P5-2: number of times this page table actually MUTATED (owner map insert,
  // PT node creation, leaf install). Under frozen placement every structure is
  // pre-built, so this must stay at 0 for the whole run -- which is exactly the
  // condition that makes concurrent access from the parallel CORE phase safe.
  // gpgpu_sim::cycle() samples it around the parallel region and aborts if it
  // moved. Counting, not preventing: the guard belongs where the phase is.
  unsigned long long m_pt_mutations = 0;
  void note_mutation() { m_pt_mutations++; }

 public:
  unsigned long long mutation_count() const { return m_pt_mutations; }

 private:
  // A1: deferred first-touch state. m_ft_buf[c] is appended only by chiplet
  // c's gmmu (pass B runs one chiplet per thread, SMs within it sequential),
  // so the buffers need no locking; the commit runs in the sequential part of
  // the cycle after the parallel passes have joined.
  bool m_defer_ft = false;
  std::vector<std::vector<uint64_t>> m_ft_buf;
  unsigned long long m_ft_commits = 0;    // buffered entries committed
  unsigned long long m_ft_dedups = 0;     // commits that inserted nothing
  unsigned long long m_ft_max_batch = 0;  // largest single-cycle batch

  // True once load_placement() succeeded: owners come from the file, never from
  // first touch.
  bool m_placement_frozen = false;
  // Page shift declared by the loaded placement map. -1 until load_placement().
  int m_placement_page_shift = -1;
  unsigned config_uniform_page_shift() const;

};

// Forward declarations only (keep dependencies minimal)
class gpgpu_sim;
class simt_core_cluster;
class memory_partition_unit;
class memory_config;  // full definition in gpu-sim.h; pointer-only use here (P3-1)

/**
 * A chiplet is a "view" over a contiguous range of global resources.
 *
 * Phase 1 rule:
 * - A chiplet does NOT own or create clusters/memory partitions.
 * - A chiplet only stores [begin, count] and pointers to the global arrays.
 *
 * This keeps the baseline code intact: the rest of GPGPU-Sim may still access
 * m_cluster[i] / m_memory_partition_unit[i] as before.
 */
class chiplet {
 public:
  chiplet(gpgpu_sim* top, unsigned chiplet_id);
  ~chiplet();

  unsigned id() const { return m_id; }

  // Bind contiguous range for clusters (global IDs)
  void bind_cluster_range(unsigned global_begin, unsigned count);

  // Bind contiguous range for memory partitions (global IDs).
  // sub_per_mem = m_n_sub_partition_per_memory_channel (SMELL-2 fix: was a
  // hardcoded 2).
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
  
  // PMM: bump allocator over this chiplet's frame range [m_pfn_begin, m_pfn_end).
  // Hands out frames in ascending order, which is exactly what the previous
  // queue-of-ascending-PFNs did, so 4KB allocation order is unchanged. Unlike a
  // queue it can also hand out contiguous runs, which huge pages require.
  // Runtime never frees (see pmm_free_page), so m_recycled only ever holds the
  // fragments skipped by huge-page alignment.
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
  // n_frames contiguous frames, aligned to n_frames. Frames skipped to reach
  // alignment are recycled for later 4KB allocations.
  uint64_t alloc_contiguous(uint64_t n_frames) {
    uint64_t aligned = (m_pfn_next + n_frames - 1) / n_frames * n_frames;
    if (aligned + n_frames > m_pfn_end) return (uint64_t)-1;
    for (uint64_t pfn = m_pfn_next; pfn < aligned; pfn++) m_recycled.push_back(pfn);
    m_pfn_next = aligned + n_frames;
    return aligned;
  }
  void free_page(uint64_t pfn) { m_recycled.push_back(pfn); }
  // s30 (-gmmu_placement_exact): record frames [pfn, pfn + n) to be taken at exactly
  // these numbers. Nothing is handed out until finalize_exact_reservations().
  // Returns false when the run leaves this chiplet's frame range.
  bool reserve_exact(uint64_t pfn, uint64_t n) {
    if (pfn < m_pfn_begin || pfn + n > m_pfn_end) return false;
    for (uint64_t k = 0; k < n; k++) m_exact_frames.push_back(pfn + k);
    return true;
  }
  // Takes every recorded frame out of the allocator: the bump cursor moves past the
  // highest one and every frame below it that is neither handed out before nor
  // recorded goes to m_recycled, so a later 4KB allocation can never collide with a
  // replayed frame. Returns false on a duplicate or on a frame already handed out
  // (e.g. the root, which the constructor allocates before the map is read).
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
    // alloc_page() pops from the back: keep ascending order of reuse = descending vector.
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

  // P5: per-chiplet L2 TLB accessors, one TLB per page size.
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

  // PMM state: bump cursor over this chiplet's local DRAM frames, plus the
  // fragments recycled from huge-page alignment.
  uint64_t m_pfn_begin = 0, m_pfn_end = 0, m_pfn_next = 0;
  std::vector<uint64_t> m_recycled;
  std::vector<uint64_t> m_exact_frames;   // s30: frames recorded by reserve_exact()
  size_t m_exact_count = 0;               // s30: frames taken by the last finalize

  // Per-chiplet L2 TLBs, owned by gpu_module, set in init() (P5). One per page size.
  gpu_l2_tlb* m_l2_tlb_4k  = nullptr;
  gpu_l2_tlb* m_l2_tlb_64k = nullptr;
  gpu_l2_tlb* m_l2_tlb_2m  = nullptr;
};

/**
 * gpu_module is the "logical GPU container" for MCM scaffolding.
 *
 * Responsibilities in Phase 1:
 * 1) Create chiplet objects.
 * 2) Decide partitioning policy: contiguous range split for clusters/mem partition.
 * 3) Build owner maps: global_id -> chiplet_id (for Phase 2 remote checks).
 * 4) Bind chiplets to global arrays once gpgpu_sim has allocated them.
 */
class gpu_module {
 public:
  gpu_module(gpgpu_sim* top, unsigned n_chiplets, unsigned n_clusters,
             unsigned n_mem, unsigned n_mem_sub_partition);
  ~gpu_module();

  /**
   * Partition global resource counts into chiplets using contiguous ranges.
   * Remainder is distributed to lower-index chiplets.
   *
   * Example: total=100, num=4 => (25,25,25,25)
   * Example: total=101, num=4 => (26,25,25,25)
   */
  void init();
  void create_chiplets();
  static void split_range(unsigned total, unsigned n_mem, unsigned part_id,
                          unsigned* out_begin, unsigned* out_count);
  /**
   * Bind pointers to global arrays.
   * This must be called after gpgpu_sim has allocated and initialized:
   * - simt_core_cluster** m_cluster
   * - memory_partition_unit** m_memory_partition_unit
   */
  void bind_global_arrays(simt_core_cluster** global_clusters,
                          memory_partition_unit** global_mem);

  // Hooking push and pop
  void icnt_push(unsigned input_deviceID, unsigned output_deviceID,
                 void* data, unsigned int size);
  void* icnt_pop(unsigned output_deviceID);
  
  // Set stats file for logging MCM access
  void set_stats_file(FILE* fout) { m_stats_file = fout; }
  
  // Set current kernel's LASP configuration
  void set_current_kernel_lasp_config(const lasp::kernel_lasp_config_t* lasp_config) { 
    m_current_lasp_config = lasp_config; 
  }
  
  // Get current kernel's LASP configuration
  const lasp::kernel_lasp_config_t* get_current_kernel_lasp_config() const { 
    return m_current_lasp_config; 
  }
  
  // Set current kernel's block dimension for memory affinity calculation
  void set_current_kernel_block_dim(dim3 block_dim) {
    m_current_kernel_block_dim = block_dim;
  }
  
  // Get current kernel's block dimension
  dim3 get_current_kernel_block_dim() const {
    return m_current_kernel_block_dim;
  }

  // Get access statistics
  // H3-2: 값은 "공유 폴백 카운터 + 모든 스레드 슬롯"의 합. unsigned 64bit 덧셈이라
  // 더하는 순서와 스레드 분할에 무관하게 결과가 정확히 같다(결합·교환법칙, 오버플로
  // 없음: 실측 최대 ~7.3e7 << 1.8e19). 시뮬레이션 중에는 읽지 않고 통계 출력에서만
  // 호출된다(gpu-sim.cc mcm_* 출력 3곳이 유일한 호출자).
  unsigned long long get_remote_access_count() const { return acc_total(0); }
  unsigned long long get_local_access_count() const { return acc_total(1); }

  // Getters
  chiplet* get_chiplet(unsigned i) const { return m_chiplets[i]; }
  unsigned get_num_chiplets() const { return m_n_chiplets; }
  // XLATDBG: which chiplet owns global memory partition `mp`.
  unsigned chiplet_of_mem(unsigned mp) const {
    return (mp < m_mem_to_chiplet.size()) ? m_mem_to_chiplet[mp] : 0;
  }

  // -gmmu_local_ptw: every chiplet owns a physical copy of the page table
  // (own root/CR3 + own node pages), so a page walk never leaves the chiplet.
  // Data pages are NOT replicated.
  bool local_ptw_enabled() const;
  // XLATDBG: times a MISS was rolled back because the PTW queue was full.
  unsigned long long m_xlatdbg_pwq_rollback = 0;

  // The single page shift this config gives EVERY allocation, or 0 if
  // -gpgpu_page_size_policy is size-dependent (policies 3-5, mixed pages).
  // Probed through decide_page_shift() so there is one source of truth.
  unsigned uniform_page_shift() const;
  unsigned get_num_clusters() const { return m_n_clusters; }
  unsigned get_num_mem() const { return m_n_mem; }
  unsigned get_num_mem_sub_partition() const { return m_n_mem_sub_partition; }
  // Provides access to memory config for GMMU child_mf creation. (P3-1)
  const memory_config* get_memory_config() const;

  // PMM interface: routes page allocation to the target chiplet. (new)
  uint64_t pmm_alloc_page(unsigned chiplet_id);
  // Allocates n_frames physically contiguous, n_frames-aligned 4KB frames.
  // Used for huge pages (2MB = GPU_HUGE_FRAMES). Returns -1 if unavailable.
  uint64_t pmm_alloc_contiguous(unsigned chiplet_id, uint64_t n_frames);
  // s30: exact-PFN replay (see chiplet::reserve_exact / finalize_exact_reservations).
  bool     pmm_reserve_exact(unsigned chiplet_id, uint64_t pfn, uint64_t n_frames);
  bool     pmm_finalize_exact(unsigned chiplet_id);
  size_t   pmm_exact_frames_taken(unsigned chiplet_id) const;

  // Page table interface.
  // Walk the page table for va. Returns the leaf PTE, or nullptr on page fault.
  const gpu_pte_t* walk_va(uint64_t va, int* walk_depth = nullptr) const;
  
  // Called when cudaMalloc/cudaMallocArray completes. (P3)
  // Registers [va_base, va_base + aligned_size) as a valid VA range.
  void on_malloc(uint64_t va_base, size_t aligned_size, unsigned page_shift);
  // Returns true if va falls within any registered cudaMalloc range. (P3)
  bool is_va_allocated(uint64_t va) const;
  // Page shift of the cudaMalloc range containing va; GPU_PAGE_SHIFT (4KB) for
  // addresses outside any registered range (local memory, params, ...).
  // Single entry point for "what page size is this address?".
  unsigned page_shift_for_va(uint64_t va) const;
  // Returns the GMMU for chiplet_id. Valid after init(). (P3)
  class gmmu* get_gmmu(unsigned chiplet_id) const;

  // Routes an L1 TLB miss through L2 TLB/MSHR, then GMMU if needed.
  // out_pte : filled when L2 TLB array hits (is_l2_hit == true).
  // is_l2_hit: true if L2 TLB array hit; caller fills L1 TLB directly.
  // Returns true  : request accepted; caller adds vpn to L1 MSHR.
  // Returns false : L2 MSHR_FULL or GMMU PWQ full; caller retries next cycle.
  bool l2_tlb_request(uint64_t va, unsigned cluster_id, uint64_t cycle,
                      gpu_pte_t& out_pte, bool& is_l2_hit,
                       bool* link_used = nullptr,
                       uint64_t* pending_until = nullptr,   // 44-3: gate expiry on LOOKUP_PENDING
                       unsigned* full_kind = nullptr);      // 44-4: 1=merge cap, 2=table full


  // Direct access to the page table
  gpu_page_table* get_page_table() const { return m_page_table; }

  void     pmm_free_page(uint64_t pfn, unsigned chiplet_id);
  unsigned pfn_to_chiplet_via_addrdec(uint64_t pa) const;

  bool is_remote_access(unsigned inputDeviceID,
                            unsigned outputDeviceID,
                            void* data) const;

  unsigned cluster_to_chiplet(unsigned global_cluster_id) const;
  unsigned mem_sub_partition_to_chiplet(unsigned global_mem_part_id) const;

  // Read-only access to the device->chiplet owner maps, for interconnect setup.
  // Sizes: cluster map == m_n_clusters, mem-sub map == m_n_mem_sub_partition.
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

  /**
   * Debug: print owner mapping.
   *
   * If sub_parts_per_mem > 0, L2 slice owner mapping is also printed, assuming:
   * - global_l2_slice_id = mem_part_id * sub_parts_per_mem + sub_part_id
   * - l2_slice owner follows the owner of its parent mem partition (Phase 1)
   */
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

  // owner maps
  // size: m_n_clusters, m_n_mem
  // value: chiplet_id
  // Given cluster_id, returns which chiplet it maps to.
  std::vector<unsigned> m_cluster_to_chiplet;
  std::vector<unsigned> m_mem_to_chiplet;
  std::vector<unsigned> m_mem_sub_partition_to_chiplet;

  // global array pointers (NOT owned)
  simt_core_cluster** m_global_clusters;
  memory_partition_unit** m_global_mem;
  
  // Stats file and counters for MCM access logging
  FILE* m_stats_file;
  // H3-2: 아래 두 개는 이제 "폴백" 누산기다. 슬롯을 못 쓰는 경우(스레드 번호가
  // 슬롯 범위를 벗어남 / OpenMP 없이 빌드 / 슬롯 0개)에만 예전처럼 atomic 으로
  // 올린다. 정상 경로는 m_acc_counts 의 스레드별 슬롯.
  unsigned long long m_remote_access_count;
  unsigned long long m_local_access_count;

  // ---- H3-2: MCM access 카운터의 스레드별 슬롯 ----------------------------
  // 왜: icnt_deliver() 는 SM->mem 요청 경로에서 매 패킷 호출되고, 예전에는 공유
  //     스칼라 2개에 #pragma omp atomic 을 걸었다. 같은 캐시라인을 모든 코어가
  //     번갈아 소유하게 되어 증가 1회가 8스레드에서 ~36ns(측정) 들었다.
  //     스레드마다 자기 라인을 쓰면 ~2.3ns 로 떨어진다.
  // 정확성: 슬롯 증가도 atomic 으로 한다. 경쟁이 없는 lock xadd 는 비용이
  //     사실상 0(측정: 평범한 증가와 동일한 2.3ns)이면서, "슬롯 하나에는 한
  //     스레드만 쓴다"는 논증에 전혀 기대지 않게 해준다. 중첩 병렬 영역이
  //     생기든, OpenMP 밖의 pthread 가 끼어들든, 스레드 번호가 겹치든
  //     카운트는 원자적으로 더해지므로 절대 유실되지 않는다.
  // 간격: 슬롯 하나가 64B(캐시라인) 를 차지하게 stride 8 -> false sharing 없음.
  enum { ACC_SLOT_STRIDE = 8, ACC_KINDS = 2 };  // [slot*8 + kind], kind 0=remote 1=local
  unsigned m_acc_slots;                      // 슬롯 개수(0 이면 폴백만 사용)
  std::vector<unsigned long long> m_acc_counts;  // 크기 = m_acc_slots * 8

  // 통계 합산. 병렬 구간에서는 호출되지 않는다(출력 시점에만).
  unsigned long long acc_total(unsigned kind) const {
    unsigned long long t = (kind == 0) ? m_remote_access_count : m_local_access_count;
    for (unsigned s = 0; s < m_acc_slots; ++s)
      t += m_acc_counts[(size_t)s * ACC_SLOT_STRIDE + kind];
    return t;
  }

  // 증가. kind 0=remote, 1=local. 정의는 gpu-module.cc 에 둔다(본문이 _OPENMP
  // 에 따라 달라지므로, -fopenmp 없이 이 헤더를 읽는 TU -- abstract_hardware_model.h
  // 를 타고 들어오는 trace-driven 쪽 -- 와 정의가 갈리지 않게 하려는 것. 호출은
  // 런당 ~60만 회뿐이라 인라인 여부는 측정에 영향이 없다).
  void bump_access_count(unsigned kind);
  
  // Current kernel's LASP configuration
  const lasp::kernel_lasp_config_t* m_current_lasp_config;
  
  // Current kernel's block dimension for memory affinity
  dim3 m_current_kernel_block_dim;

  gpu_page_table* m_page_table;  // single GPU-wide page table (node PAs get
                                 // per-chiplet copies under -gmmu_local_ptw)
  
  std::vector<va_alloc_t> m_va_alloc_list;  // VA ranges from all cudaMalloc calls (P3)
  // P-6 (speed-audit V4 residue): uniform_page_shift() cached at init. Nonzero
  // under a FIXED page-size policy, where page_shift_for_va() provably answers
  // this value for EVERY va (registered allocations all carry it, and the
  // out-of-range fallback returns it too) -- so the hot allocation-list scan
  // can be skipped entirely. 0 under automatic policies (3-5): scan required.
  unsigned m_uniform_shift = 0;
  std::vector<class gmmu*> m_gmmus;          // one GMMU per chiplet, indexed by chiplet_id (P3)
  
  // C-4 (SMELL-11): the legacy fixed-latency inter-chiplet delay queue was
  // removed -- inter-chiplet latency and bandwidth are modeled solely by the
  // ring (-icnt_use_ring / -icnt_ring_hop_latency). init() aborts if the old
  // -icnt_inter_hop_latency_ns option is set nonzero.
  void icnt_deliver(unsigned in, unsigned out, void* data, unsigned size);

};

#endif  // GPU_MODULE_H
