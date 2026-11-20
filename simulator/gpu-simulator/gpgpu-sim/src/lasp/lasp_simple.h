#ifndef LASP_SIMPLE_H
#define LASP_SIMPLE_H

#include <string>
#include <vector>
#include <cstddef>

namespace lasp {

// LASP CTA -> chiplet binding (formulas as in simt_core_cluster::issue_block2core()).
// -gpgpu_lasp_policy 1 = ROW, 2 = COL, 3 = CHUNK, 4 = KERNEL_PARTITION (CHUNK, chunk = ceil(CTAs / chiplets)).
enum scheduling_policy_t {
  ROW_BINDING = 0,    // chiplet = TB_y / ceil(grid_y / chiplets)
  COL_BINDING = 1,    // chiplet = TB_x / ceil(grid_x / chiplets)
  CHUNK_BINDING = 2,  // chiplet = (linear TB id / chunk_size) % chiplets
  KERNEL_WIDE = 3     // no binding (pre-LASP issue path)
};

// Per-kernel LASP setting.
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

// Policy name for logs.
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
