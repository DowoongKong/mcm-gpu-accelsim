#ifndef LASP_HEURISTIC_H
#define LASP_HEURISTIC_H

#include "lasp_simple.h"
#include <vector_types.h>

// Forward declaration
class kernel_info_t;

// ============================================================================
// LASP Heuristic Analyzer
// ============================================================================
// Analyzes kernel characteristics (grid size, block size, kernel name)
// to automatically determine the best CTA scheduling policy.
//
// No config file needed - the heuristic is applied at runtime based on
// the actual kernel properties.
//

class lasp_heuristic {
 public:
  // Main function: Analyze kernel and return recommended policy
  // 
  // Args:
  //   kernel: kernel_info_t with grid_dim, block_dim, name
  //   num_chiplets: Number of chiplets in the GPU
  //
  // Returns:
  //   kernel_lasp_config_t with:
  //   - policy: Automatically determined policy
  //   - chunk_size: If CHUNK_BINDING, appropriate chunk size
  //
  static lasp::kernel_lasp_config_t analyze_kernel(
      kernel_info_t *kernel,
      unsigned num_chiplets);

 private:
  // ========================================================================
  // Heuristic Rules
  // ========================================================================

  // Rule 1: Grid shape analysis
  // - Square or near-square grid → ROW_BINDING (good row locality)
  // - Linear grid (very small y-dimension) → CHUNK_BINDING (sequential data)
  // - Irregular shape → analyze further
  static bool is_square_grid(dim3 grid_dim);
  static bool is_linear_grid(dim3 grid_dim);
  
  // Rule 2: Block size analysis
  // - Large block (>512 threads) → likely complex kernel, use ROW_BINDING
  // - Small block (<256 threads) → simple kernel, use CHUNK_BINDING
  static bool is_large_block(dim3 block_dim);
  
  // Rule 3: Kernel name heuristics
  // Some common names suggest specific patterns
  static lasp::scheduling_policy_t guess_from_name(const std::string &name);
  
  // Rule 4: Calculate appropriate chunk size for CHUNK_BINDING
  static unsigned calculate_chunk_size(
      dim3 grid_dim,
      unsigned num_chiplets);
};

#endif // LASP_HEURISTIC_H
