#ifndef LASP_SIMPLE_H
#define LASP_SIMPLE_H

#include <string>
#include <vector>
#include <cstddef>

namespace lasp {

// ============================================================================
// Scheduling Policy Enum
// ============================================================================
// Defines the CTA scheduling strategies for LASP
//
// Formulas below are the LIVE implementation's (simt_core_cluster::
// issue_block2core); an earlier version of this comment had ROW/COL swapped.
//
// ROW_BINDING:
//   - Same y-coordinate CTAs are assigned to the same chiplet
//   - Use case: row-major data access (e.g., GEMM matrix A)
//   - chiplet = TB_y / ceil(grid_y / num_chiplets)   (contiguous row blocks)
//
// COL_BINDING:
//   - Same x-coordinate CTAs are assigned to the same chiplet
//   - Use case: column-major data access (e.g., GEMM matrix B)
//   - chiplet = TB_x / ceil(grid_x / num_chiplets)   (contiguous column blocks)
//
// CHUNK_BINDING:
//   - chunk_size consecutive linear CTA ids round-robin across chiplets
//   - Use case: strided/tiled access; chunk_size=1 is plain round-robin
//   - chiplet = (linear_TB_id / chunk_size) % num_chiplets
//
// KERNEL_WIDE:
//   - Default: NO binding; the pre-LASP issue path runs unchanged
//
enum scheduling_policy_t {
  ROW_BINDING = 0,
  COL_BINDING = 1,
  CHUNK_BINDING = 2,
  KERNEL_WIDE = 3
};

// ============================================================================
// Per-Kernel LASP Configuration Structure
// ============================================================================
// Stores the scheduling policy and data information for a kernel
//
struct kernel_lasp_config_t {
  // Scheduling policy to apply
  scheduling_policy_t policy;
  
  // Chunk size for CHUNK_BINDING policy
  // (ignored for other policies)
  unsigned chunk_size;
  
  // Constructor with default values
  kernel_lasp_config_t()
      : policy(KERNEL_WIDE), chunk_size(1) {}
};

// ============================================================================
// Helper Functions (optional, for future use)
// ============================================================================

// Convert policy enum to string for logging
inline const char* policy_to_string(scheduling_policy_t policy) {
  switch (policy) {
    case ROW_BINDING:    return "ROW_BINDING";
    case COL_BINDING:    return "COL_BINDING";
    case CHUNK_BINDING:  return "CHUNK_BINDING";
    case KERNEL_WIDE:    return "KERNEL_WIDE";
    default:             return "UNKNOWN";
  }
}

} // namespace lasp

#endif // LASP_SIMPLE_H
