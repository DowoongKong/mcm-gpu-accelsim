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
// ROW_BINDING:
//   - Same y-coordinate CTAs are assigned to the same chiplet
//   - Use case: Row-major data access (e.g., GEMM matrix A)
//   - Formula: chiplet_id = TB_x / (grid_x / num_chiplets)
//
// COL_BINDING:
//   - Same x-coordinate CTAs are assigned to the same chiplet
//   - Use case: Column-major data access (e.g., GEMM matrix B)
//   - Formula: chiplet_id = TB_y / (grid_y / num_chiplets)
//
// CHUNK_BINDING:
//   - CTAs are grouped by chunk_size and round-robin across chiplets
//   - Use case: Strided/tiled data access patterns
//   - Formula: chiplet_id = (TB_id / chunk_size) % num_chiplets
//
// KERNEL_WIDE:
//   - Default: uniform distribution (existing behavior)
//   - Use case: Unpredictable memory access patterns
//   - Formula: chiplet_id = TB_id / (total_ctas / num_chiplets)
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
