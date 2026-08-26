// LASP Scheduler Implementation
// Policy-specific CTA coordinate calculation

#include "lasp_scheduler.h"
#include <cstdio>
#include <cassert>

namespace lasp_scheduler {

/**
 * Main dispatcher that selects appropriate scheduling function based on policy
 */
dim3 get_next_cta_id_lasp(kernel_info_t *kernel,
                          lasp::scheduling_policy_t policy,
                          unsigned chunk_size,
                          unsigned num_chiplets) {
  // Sanity checks
  assert(kernel != NULL);
  assert(num_chiplets > 0);
  
  dim3 grid_dim = kernel->get_grid_dim();
  assert(grid_dim.x > 0 && grid_dim.y > 0 && grid_dim.z > 0);
  
  switch (policy) {
    case lasp::ROW_BINDING:
      return schedule_row_binding(kernel, num_chiplets);
    
    case lasp::COL_BINDING:
      return schedule_col_binding(kernel, num_chiplets);
    
    case lasp::CHUNK_BINDING:
      return schedule_chunk_binding(kernel, chunk_size, num_chiplets);
    
    case lasp::KERNEL_WIDE:
      return schedule_kernel_wide(kernel, num_chiplets);
    
    default:
      // Fallback to kernel-wide if unknown policy
      return schedule_kernel_wide(kernel, num_chiplets);
  }
}

/**
 * ROW_BINDING Implementation
 * 
 * Goal: Keep CTAs with same Y-coordinate on same chiplet
 * This preserves row-wise data access locality
 * 
 * Algorithm:
 * 1. Get current CTA index (sequential)
 * 2. Calculate 2D coordinates (tb_x, tb_y) from linear index
 * 3. Determine chiplet assignment from Y coordinate: 
 *    chiplet = TB_y / (grid_y / num_chiplets)
 * 4. Rotate CTA allocation to match chiplet layout
 * 
 * Example: 10x10 grid, 4 chiplets (2x2 layout)
 * TB_y in [0,4]: chiplet 0,1 (top)
 * TB_y in [5,9]: chiplet 2,3 (bottom)
 */
dim3 schedule_row_binding(kernel_info_t *kernel, unsigned num_chiplets) {
  dim3 grid_dim = kernel->get_grid_dim();
  unsigned grid_x = grid_dim.x;
  unsigned grid_y = grid_dim.y;
  unsigned grid_z = grid_dim.z;
  
  // Get current next CTA ID (this is what we're about to allocate)
  dim3 current_cta = kernel->get_next_cta_id();
  unsigned tb_x = current_cta.x;
  unsigned tb_y = current_cta.y;
  unsigned tb_z = current_cta.z;
  
  // Calculate optimal chiplet for this TB based on row
  unsigned rows_per_chiplet = (grid_y + num_chiplets - 1) / num_chiplets;
  unsigned target_chiplet = tb_y / rows_per_chiplet;
  if (target_chiplet >= num_chiplets) {
    target_chiplet = num_chiplets - 1;
  }
  
  // For now, return sequential coordinates
  // Actual chiplet assignment happens at GPU simulator level
  return dim3(tb_x, tb_y, tb_z);
}

/**
 * COL_BINDING Implementation
 * 
 * Goal: Keep CTAs with same X-coordinate on same chiplet
 * This preserves column-wise data access locality
 * 
 * Algorithm:
 * 1. Get current CTA index (sequential)
 * 2. Calculate 2D coordinates (tb_x, tb_y) from linear index
 * 3. Determine chiplet assignment from X coordinate:
 *    chiplet = TB_x / (grid_x / num_chiplets)
 * 4. May need to interleave allocation for better distribution
 * 
 * Example: 10x10 grid, 4 chiplets (2x2 layout)
 * TB_x in [0,4]: chiplet 0,2 (left)
 * TB_x in [5,9]: chiplet 1,3 (right)
 */
dim3 schedule_col_binding(kernel_info_t *kernel, unsigned num_chiplets) {
  dim3 grid_dim = kernel->get_grid_dim();
  unsigned grid_x = grid_dim.x;
  unsigned grid_y = grid_dim.y;
  unsigned grid_z = grid_dim.z;
  
  // Get current next CTA ID
  dim3 current_cta = kernel->get_next_cta_id();
  unsigned tb_x = current_cta.x;
  unsigned tb_y = current_cta.y;
  unsigned tb_z = current_cta.z;
  
  // Calculate optimal chiplet for this TB based on column
  unsigned cols_per_chiplet = (grid_x + num_chiplets - 1) / num_chiplets;
  unsigned target_chiplet = tb_x / cols_per_chiplet;
  if (target_chiplet >= num_chiplets) {
    target_chiplet = num_chiplets - 1;
  }
  
  // For now, return sequential coordinates
  // Actual chiplet assignment happens at GPU simulator level
  return dim3(tb_x, tb_y, tb_z);
}

/**
 * CHUNK_BINDING Implementation
 * 
 * Goal: Assign CTAs in fixed-size chunks round-robin across chiplets
 * Useful when kernel has unclear locality patterns
 * 
 * Algorithm:
 * 1. Get current CTA index (sequential)
 * 2. Determine which chunk: chunk_id = cta_index / chunk_size
 * 3. Assign chiplet: chiplet = chunk_id % num_chiplets
 * 4. Rearrange position within chunk for better memory coalescing
 * 
 * Example: 256 CTAs, chunk_size=64, 4 chiplets
 * CTAs [0-63]   -> Chiplet 0
 * CTAs [64-127] -> Chiplet 1
 * CTAs [128-191]-> Chiplet 2
 * CTAs [192-255]-> Chiplet 3
 * 
 * Then within each chunk, distribute Y-coordinates evenly
 */
dim3 schedule_chunk_binding(kernel_info_t *kernel,
                            unsigned chunk_size,
                            unsigned num_chiplets) {
  dim3 grid_dim = kernel->get_grid_dim();
  unsigned grid_x = grid_dim.x;
  unsigned grid_y = grid_dim.y;
  unsigned grid_z = grid_dim.z;
  unsigned total_ctas = grid_x * grid_y * grid_z;
  
  // Validate chunk_size
  if (chunk_size == 0) {
    chunk_size = 1;
  }
  if (chunk_size > total_ctas) {
    chunk_size = total_ctas;
  }
  
  // Get current next CTA ID
  dim3 current_cta = kernel->get_next_cta_id();
  unsigned tb_x = current_cta.x;
  unsigned tb_y = current_cta.y;
  unsigned tb_z = current_cta.z;
  
  // Convert 3D to linear index
  unsigned cta_index = tb_x + tb_y * grid_x + tb_z * grid_x * grid_y;
  
  // Determine chunk and target chiplet
  unsigned chunk_id = cta_index / chunk_size;
  unsigned target_chiplet = chunk_id % num_chiplets;
  
  // For now, return sequential coordinates
  // Actual chiplet assignment happens at GPU simulator level
  return dim3(tb_x, tb_y, tb_z);
}

/**
 * KERNEL_WIDE Implementation (Default/Fallback)
 * 
 * Goal: Uniformly distribute CTAs across all chiplets
 * Safe choice when kernel characteristics are unknown
 * 
 * Algorithm:
 * 1. Get current CTA index (sequential)
 * 2. Assign chiplet: chiplet = cta_index % num_chiplets
 * 3. Return sequential 3D coordinates (distribution happens at GPU level)
 * 
 * This is essentially the default behavior - sequential allocation
 * with round-robin chiplet assignment at GPU simulator level
 */
dim3 schedule_kernel_wide(kernel_info_t *kernel, unsigned num_chiplets) {
  dim3 grid_dim = kernel->get_grid_dim();
  unsigned grid_x = grid_dim.x;
  unsigned grid_y = grid_dim.y;
  unsigned grid_z = grid_dim.z;
  
  // Get current next CTA ID
  dim3 current_cta = kernel->get_next_cta_id();
  unsigned tb_x = current_cta.x;
  unsigned tb_y = current_cta.y;
  unsigned tb_z = current_cta.z;
  
  // Convert 3D to linear index
  unsigned cta_index = tb_x + tb_y * grid_x + tb_z * grid_x * grid_y;
  
  // Determine target chiplet for uniform distribution
  unsigned target_chiplet = cta_index % num_chiplets;
  
  // For now, return sequential coordinates
  // Actual chiplet assignment happens at GPU simulator level
  return dim3(tb_x, tb_y, tb_z);
}

}  // namespace lasp_scheduler
