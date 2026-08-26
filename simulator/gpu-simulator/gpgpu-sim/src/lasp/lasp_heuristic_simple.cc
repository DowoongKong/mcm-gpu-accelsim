#include "lasp_heuristic.h"
#include "../abstract_hardware_model.h"
#include <cstring>
#include <stdio.h>

// Simplified LASP analyzer - just always return KERNEL_WIDE for now
lasp::kernel_lasp_config_t lasp_heuristic::analyze_kernel(
    kernel_info_t *kernel,
    unsigned num_chiplets)
{
  fprintf(stdout, "[LASP] analyze_kernel called for kernel '%s'\n", kernel->name().c_str());
  fflush(stdout);
  
  lasp::kernel_lasp_config_t config;
  config.kernel_name = kernel->name();
  config.policy = lasp::KERNEL_WIDE;
  config.chunk_size = 1;
  
  fprintf(stdout, "[LASP] Selected policy: %s\n", 
          lasp::policy_to_string(config.policy));
  fflush(stdout);
  
  return config;
}
