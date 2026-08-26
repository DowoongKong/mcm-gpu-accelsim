#include "lasp_heuristic.h"
#include "../abstract_hardware_model.h"
#include <cstring>
#include <stdio.h>

lasp::kernel_lasp_config_t lasp_heuristic::analyze_kernel(
    kernel_info_t *kernel,
    unsigned num_chiplets)
{
  fprintf(stdout, "[LASP] analyze_kernel called\n");
  fflush(stdout);
  
  lasp::kernel_lasp_config_t config;
  config.policy = lasp::CHUNK_BINDING;  // Use CHUNK_BINDING for round-robin distribution across chiplets
  config.chunk_size = 1;  // 1 CTA per chunk = round-robin
  
  fprintf(stdout, "[LASP] Selected policy: %s\n", 
          lasp::policy_to_string(config.policy));
  fflush(stdout);
  
  return config;
}

bool lasp_heuristic::is_square_grid(dim3 grid_dim) { return false; }
bool lasp_heuristic::is_linear_grid(dim3 grid_dim) { return false; }
bool lasp_heuristic::is_large_block(dim3 block_dim) { return false; }
lasp::scheduling_policy_t lasp_heuristic::guess_from_name(const std::string &name) { return lasp::KERNEL_WIDE; }
unsigned lasp_heuristic::calculate_chunk_size(dim3 grid_dim, unsigned num_chiplets) { return 1; }
