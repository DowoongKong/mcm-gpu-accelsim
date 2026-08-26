// LASP Scheduler: Policy-specific CTA scheduling functions
// Implements ROW_BINDING, COL_BINDING, CHUNK_BINDING scheduling

#ifndef __LASP_SCHEDULER_H__
#define __LASP_SCHEDULER_H__

#include "lasp_simple.h"
#include "../abstract_hardware_model.h"

namespace lasp_scheduler {

/**
 * Calculate next CTA coordinates based on LASP scheduling policy
 * 
 * @param kernel          kernel_info_t object with grid/block dimensions
 * @param policy          LASP scheduling policy (ROW_BINDING, etc.)
 * @param chunk_size      Size of chunk for CHUNK_BINDING policy
 * @param num_chiplets    Total number of chiplets in MCM-GPU
 * @return dim3 with next CTA (x, y, z) coordinates
 */
dim3 get_next_cta_id_lasp(kernel_info_t *kernel,
                          lasp::scheduling_policy_t policy,
                          unsigned chunk_size,
                          unsigned num_chiplets);

/**
 * ROW_BINDING: Same Y-coordinate CTAs are assigned to same chiplet
 * Maps TB_x / (grid_x / num_chiplets) to chiplet
 * This keeps CTAs with same Y together for better reuse
 * Useful for kernels with row-major data access patterns
 */
dim3 schedule_row_binding(kernel_info_t *kernel,
                          unsigned num_chiplets);

/**
 * COL_BINDING: Same X-coordinate CTAs are assigned to same chiplet
 * Maps TB_y / (grid_y / num_chiplets) to chiplet
 * This keeps CTAs with same X together for better locality
 * Useful for kernels with column-major data access patterns
 */
dim3 schedule_col_binding(kernel_info_t *kernel,
                          unsigned num_chiplets);

/**
 * CHUNK_BINDING: CTAs grouped by chunk_size, round-robin across chiplets
 * Assigns CTAs in groups of chunk_size to each chiplet sequentially
 * Example: 256 CTAs, 4 chiplets, chunk=64 -> [0-63]->C0, [64-127]->C1, etc.
 * Useful for linear grids or workloads where exact locality is unknown
 */
dim3 schedule_chunk_binding(kernel_info_t *kernel,
                            unsigned chunk_size,
                            unsigned num_chiplets);

/**
 * KERNEL_WIDE: Uniform distribution across chiplets (default/fallback)
 * Simply round-robin CTAs across all chiplets
 * Safe choice when kernel characteristics are unknown
 */
dim3 schedule_kernel_wide(kernel_info_t *kernel,
                          unsigned num_chiplets);

}  // namespace lasp_scheduler

#endif  // __LASP_SCHEDULER_H__
