// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifdef _OPENMP
#include <omp.h>
#endif
#include "gpu-sim.h"

#include <algorithm>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"
#include "gpu-module.h"  // MCM chiplet topology wrapper
#include "gmmu.h"        // full definition needed: gmmu->cycle() is called below
#include "par_serial_probe.h"
#include "../lasp/lasp_simple.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"

#include <chrono>  // PHASE_TIMING: wall-clock phase-boundary instrumentation

// ICNT cycle stamp for the ACCELSIM_DIAG reply-latency rows; defined in local_interconnect.cc.
extern unsigned long long g_icnt_now;

// [ACCEPT] (ACCELSIM_DIAG): per-chiplet L2-loop visits, all (total) and with a full input (stall).
unsigned long long g_memfull_stall[8] = {0};
unsigned long long g_memfull_total[8] = {0};

// ACCELSIM_PHASE_TIMING: wall-clock time per cycle() block, printed to stderr ([PHASE_TIMING*]).
// extern is required: a const global has internal linkage, and local_interconnect.cc reads it.
extern const bool g_phase_timing_on = (getenv("ACCELSIM_PHASE_TIMING") != NULL);

// ACCELSIM_DIAG: turns on the diagnostic counters and rows ([ACCEPT], [POPGATE], [NORMREP], ...).
// Off by default; the translation statistics always print.
extern const bool g_accelsim_diag = (getenv("ACCELSIM_DIAG") != NULL);

// ACCELSIM_LOG_CTA: print [CTA_ISSUE] rows; each pairs with the next [CTA_DONE] on (sm, hw_cta).
static bool accelsim_log_cta() {
  static const bool on = (getenv("ACCELSIM_LOG_CTA") != NULL);
  return on;
}
struct phase_timers_t {
  unsigned long long dram_ns = 0, l2gmmu_ns = 0, icnt_ns = 0;
  unsigned long long cycle_ns = 0, cluster_icnt_ns = 0, tail_ns = 0;
  unsigned long long tail_issue_ns = 0, tail_occ_ns = 0;
  unsigned long long phase_a_ns = 0, phase_b_ns = 0, phase_c_ns = 0;
  unsigned long long dram_calls = 0, l2gmmu_calls = 0, icnt_calls = 0;
  unsigned long long phase_a_calls = 0, phase_b_calls = 0, phase_c_calls = 0;
  // [PHASE_TIMING_B]: phase B time and running thread per chiplet group (one writer per index).
  unsigned long long phase_b_chip_ns[8] = {0};
  unsigned long long phase_b_chip_thread[8][16] = {{0}};
  // [PHASE_TIMING_B2]: per-cycle group times and their maximum (sum, argmax, log2(us) histograms).
  unsigned long long phase_b_chip_cur_ns[8] = {0};
  unsigned long long phase_b_cyclemax_ns = 0;
  unsigned long long phase_b_argmax_cnt[8] = {0};
  unsigned long long phase_b_max_hist[24] = {0};
  unsigned long long phase_b_body_hist[24] = {0};
};
static inline unsigned phase_log2us_bucket(unsigned long long ns) {  // log2(us), 0..23
  unsigned long long us = ns / 1000ull;
  unsigned b = 0;
  while (us > 1 && b < 23) { us >>= 1; b++; }
  return b;
}
static phase_timers_t g_phase_timers;
// icnt_transfer() time: per-chiplet Advance() (parallel) vs. serial handoff between chiplets.
// Plain globals so local_interconnect.cc can extern them.
unsigned long long g_icnt_local_ns = 0, g_icnt_handoff_ns = 0;
unsigned long long g_icnt_local_calls = 0, g_icnt_handoff_calls = 0;
// Time of the L2 -> ICNT reply loop alone; icnt_ns also includes icnt_transfer().
unsigned long long g_icnt_pop_ns = 0, g_icnt_pop_calls = 0;
static inline unsigned long long phase_now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"
#include "../lasp/lasp_simple.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

// #define MAX(a, b) (((a) > (b)) ? (a) : (b)) //redefined

bool g_interactive_debugger_enabled = false;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name,
                         "Hardware Performance Statistics file", "hw_perf.csv");

  option_parser_register(
      opp, "-hw_perf_bench_name", OPT_CSTR, &g_hw_perf_bench_name,
      "Kernel Name in Hardware Performance Statistics file", "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power "
                         "simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)",
                         "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL, &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  // Accelwattch Hyrbid Configuration

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RH],
      "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RM],
      "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WH],
      "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WM],
      "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RH],
      "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RM],
      "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WH],
      "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WM],
      "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CC_ACC],
      "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_SHRD_ACC],
      "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim",
                         "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_DRAM_WR],
      "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NOC],
      "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
      "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim",
      "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
      "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CYCLES],
      "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_VOLTAGE],
      "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");

  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_frfcfs_age_cap", OPT_UINT32,
                         &gpgpu_frfcfs_age_cap,
                         "FR-FCFS starvation cap: once a bank's oldest request has waited this "
                         "many core cycles, the bank switches to its row instead of serving more "
                         "row hits (0 = no cap)",
                         "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  option_parser_register(opp, "-icnt_inter_flit_size", OPT_UINT32, &icnt_inter_flit_size,
                         "flit size (bytes) of inter-chiplet ring links", "8");
  // SST mode activate
  option_parser_register(opp, "-SST_mode", OPT_BOOL, &SST_mode, "SST mode",
                         "0");
  m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  // L1 TLB configuration options 
  option_parser_register(opp, "-gpgpu_l1_tlb_4k_sets", OPT_UINT32,
                         &gpgpu_l1_tlb_4k_sets,
                         "L1 TLB (4KB) sets: 1 = fully-associative", "1");
  option_parser_register(opp, "-gpgpu_l1_tlb_4k_assoc", OPT_UINT32,
                         &gpgpu_l1_tlb_4k_assoc,
                         "L1 TLB (4KB) ways (entries when fully-associative)", "32");
  option_parser_register(opp, "-gpgpu_l1_tlb_4k_mshr_entries", OPT_UINT32,
                         &gpgpu_l1_tlb_4k_mshr_entries,
                         "L1 TLB (4KB) MSHR in-flight VPN capacity", "16");
  option_parser_register(opp, "-gpgpu_l1_tlb_4k_mshr_max_merge", OPT_UINT32,
                         &gpgpu_l1_tlb_4k_mshr_max_merge,
                         "L1 TLB (4KB) max merged requests per in-flight VPN", "192");
  option_parser_register(opp, "-gpgpu_l1_tlb_4k_lookup_latency", OPT_UINT32,
                         &gpgpu_l1_tlb_4k_lookup_latency,
                         "L1 TLB (4KB) lookup latency in core cycles", "10");
  // L1 TLB (64KB) options; idle unless -gpgpu_page_size_policy uses 64KB pages.
  // Sizes follow the MCM-GPU reference config; MSHR equals the 4KB TLB's so only page size differs.
  option_parser_register(opp, "-gpgpu_l1_tlb_64k_sets", OPT_UINT32,
                         &gpgpu_l1_tlb_64k_sets,
                         "L1 TLB (64KB) sets: 1 = fully-associative", "1");
  option_parser_register(opp, "-gpgpu_l1_tlb_64k_assoc", OPT_UINT32,
                         &gpgpu_l1_tlb_64k_assoc,
                         "L1 TLB (64KB) ways (entries when fully-associative)", "16");
  option_parser_register(opp, "-gpgpu_l1_tlb_64k_mshr_entries", OPT_UINT32,
                         &gpgpu_l1_tlb_64k_mshr_entries,
                         "L1 TLB (64KB) MSHR in-flight VPN capacity", "16");
  option_parser_register(opp, "-gpgpu_l1_tlb_64k_mshr_max_merge", OPT_UINT32,
                         &gpgpu_l1_tlb_64k_mshr_max_merge,
                         "L1 TLB (64KB) max merged requests per in-flight VPN", "192");
  option_parser_register(opp, "-gpgpu_l1_tlb_64k_lookup_latency", OPT_UINT32,
                         &gpgpu_l1_tlb_64k_lookup_latency,
                         "L1 TLB (64KB) lookup latency in core cycles", "10");
  // L1 TLB (2MB) options; idle unless -gpgpu_page_size_policy uses 2MB pages.
  // Sizes follow the MCM-GPU reference config; MSHR equals the 4KB TLB's so only page size differs.
  option_parser_register(opp, "-gpgpu_l1_tlb_2m_sets", OPT_UINT32,
                         &gpgpu_l1_tlb_2m_sets,
                         "L1 TLB (2MB) sets: 1 = fully-associative", "1");
  option_parser_register(opp, "-gpgpu_l1_tlb_2m_assoc", OPT_UINT32,
                         &gpgpu_l1_tlb_2m_assoc,
                         "L1 TLB (2MB) ways (entries when fully-associative)", "8");
  option_parser_register(opp, "-gpgpu_l1_tlb_2m_mshr_entries", OPT_UINT32,
                         &gpgpu_l1_tlb_2m_mshr_entries,
                         "L1 TLB (2MB) MSHR in-flight VPN capacity", "16");
  option_parser_register(opp, "-gpgpu_l1_tlb_2m_mshr_max_merge", OPT_UINT32,
                         &gpgpu_l1_tlb_2m_mshr_max_merge,
                         "L1 TLB (2MB) max merged requests per in-flight VPN", "192");
  option_parser_register(opp, "-gpgpu_l1_tlb_2m_lookup_latency", OPT_UINT32,
                         &gpgpu_l1_tlb_2m_lookup_latency,
                         "L1 TLB (2MB) lookup latency in core cycles", "10");
  option_parser_register(opp, "-gpgpu_page_size_policy", OPT_UINT32,
                         &gpgpu_page_size_policy,
                         "page size policy. fixed (every alloc same size): "
                         "0 = all 4KB (baseline), 1 = all 64KB, 2 = all 2MB. "
                         "automatic (largest page with alloc size >= page size): "
                         "3 = 4KB/64KB/2MB, 4 = 4KB/2MB, 5 = 4KB/64KB", "0");
  option_parser_register(opp, "-gpgpu_l2_tlb_issue_width", OPT_UINT32,
                         &gpgpu_l2_tlb_issue_width,
                         "L1->L2 TLB requests one SM may issue per cycle. Issue is "
                         "decoupled from completion, so this is the interface width, "
                         "not the number of outstanding lookups (that is the MSHR).",
                         "2");
  option_parser_register(opp, "-gpgpu_l1_tlb_ports", OPT_UINT32,
                         &gpgpu_l1_tlb_ports,
                         "concurrent array probes the per-SM L1 TLB can run "
                         "(0 = unlimited, the pre-existing behaviour)",
                         "0");  // off by default: occupancy ports inflate cycle counts
  option_parser_register(opp, "-gpgpu_l1_tlb_lookups_per_cycle", OPT_UINT32,
                         &gpgpu_l1_tlb_lookups_per_cycle,
                         "new lookups each L1 TLB starts per cycle (pipelined "
                         "ports; each still takes the full lookup latency). "
                         "Requests beyond it wait in the TLB's input queue. "
                         "0 = unlimited",
                         "0");
  option_parser_register(opp, "-gpgpu_l2_tlb_ports", OPT_UINT32,
                         &gpgpu_l2_tlb_ports,
                         "concurrent array probes the per-chiplet L2 TLB can run "
                         "(0 = unlimited, the pre-existing behaviour)",
                         "0");  // off by default, see above; the L2 gate distorts more than the L1 one
  option_parser_register(opp, "-gpgpu_l2_tlb_starts_per_cycle", OPT_UINT32,
                         &gpgpu_l2_tlb_starts_per_cycle,
                         "new lookups each per-chiplet L2 TLB starts per cycle "
                         "(pipelined; each still takes the full lookup latency). "
                         "Refused requests retry the next cycle, and the SM that "
                         "goes first rotates every cycle. 0 = unlimited",
                         "0");
  option_parser_register(opp, "-gpgpu_ldst_replay_first", OPT_BOOL,
                         &gpgpu_ldst_replay_first,
                         "a LSU dispatch slot freed in a cycle goes to a parked "
                         "memory instruction whose translation completed before "
                         "a new memory instruction (0 = the new instruction "
                         "first; a replay enters only a slot empty at cycle start)",
                         "0");
  option_parser_register(opp, "-gpgpu_ldst_bypass_send_per_bank", OPT_BOOL,
                         &gpgpu_ldst_bypass_send_per_bank,
                         "the L1D-bypassing accesses of one memory instruction "
                         "are sent up to -gpgpu_l1_banks per cycle, as in "
                         "Accel-Sim 2.0 (0 = one per cycle)",
                         "0");
  option_parser_register(opp, "-gpgpu_ldst_banked_response", OPT_BOOL,
                         &gpgpu_ldst_banked_response,
                         "the ldst unit takes up to -gpgpu_l1_banks responses "
                         "per cycle and writes back up to -gpgpu_l1_banks "
                         "L1D-bypassing responses in one cycle, as in Accel-Sim "
                         "2.0 (0 = one response per cycle through a one-entry "
                         "stage)",
                         "0");
  option_parser_register(opp, "-gpgpu_alu_ready_queue", OPT_BOOL,
                         &gpgpu_alu_ready_queue,
                         "pipelined ALU units keep issued instructions in an "
                         "in-order queue, ready latency - initiation interval "
                         "cycles after dispatch, leaving only into a free result "
                         "slot, as in Accel-Sim 2.0 (0 = shift register, one "
                         "cycle later when latency > interval, with a result "
                         "slot reserved at issue)",
                         "0");
  option_parser_register(opp, "-gpgpu_ldst_gate_skip_shared", OPT_BOOL,
                         &gpgpu_ldst_gate_skip_shared,
                         "shared-memory LOAD/STORE do not take the per-warp ldst "
                         "gate (they need no translation)",
                         "0");
  option_parser_register(opp, "-gpgpu_ldst_gate_release_at_xlat", OPT_BOOL,
                         &gpgpu_ldst_gate_release_at_xlat,
                         "a LOAD/STORE releases the per-warp ldst gate once every "
                         "page it touches is translated, instead of when it "
                         "leaves the ldst dispatch",
                         "0");
  option_parser_register(opp, "-gpgpu_l1_tlb_mshr_wait_fifo", OPT_BOOL,
                         &gpgpu_l1_tlb_mshr_wait_fifo,
                         "L1 TLB lookups blocked by a full MSHR retry oldest first "
                         "(0 = the most recently blocked first)",
                         "0");
  option_parser_register(opp, "-gpgpu_l1_tlb_mshr_wait_age", OPT_BOOL,
                         &gpgpu_l1_tlb_mshr_wait_age,
                         "L1 TLB lookups blocked by a full MSHR take freed entries before new "
                         "lookups, oldest instruction first (a warp's lookups kept together); "
                         "cannot be combined with -gpgpu_l1_tlb_mshr_wait_fifo",
                         "0");
  option_parser_register(opp, "-gpgpu_shmem_per_cta_reserve", OPT_UINT32,
                         &gpgpu_shmem_per_cta_reserve,
                         "shared memory bytes the driver reserves per CTA of a kernel that uses "
                         "shared memory (1024 from sm_80); counted in the CTA-per-SM limit and the "
                         "L1D carveout choice (0 = not modeled)",
                         "0");
  option_parser_register(opp, "-gpgpu_l2_tlb_mshr_age_order", OPT_BOOL,
                         &gpgpu_l2_tlb_mshr_age_order,
                         "a free L2 TLB MSHR entry goes to the request that has waited longest "
                         "for one (first blocked cycle), whatever the SM order (0 = first asker "
                         "in SM order)",
                         "0");
  option_parser_register(opp, "-gpgpu_trace_atomic_no_stall", OPT_BOOL,
                         &gpgpu_trace_atomic_no_stall,
                         "a warp keeps issuing while its atomics are in flight; only "
                         "readers of an atomic's result wait, and the warp exits after "
                         "every reply (0 = the warp issues nothing until all replies "
                         "are back, a PTX functional-mode stall)",
                         "0");
  option_parser_register(opp, "-gpgpu_trace_rz_no_reg", OPT_BOOL,
                         &gpgpu_trace_rz_no_reg,
                         "R255 (RZ, the zero register) is not a register operand: no "
                         "scoreboard entry and no register bank read (0 = register 256)",
                         "0");
  option_parser_register(opp, "-gpgpu_membar_wait_stores", OPT_BOOL,
                         &gpgpu_membar_wait_stores,
                         "MEMBAR waits for the warp's store acks as well as its "
                         "pending register writes (0 = register writes only)",
                         "0");
  option_parser_register(opp, "-gpgpu_strong_store_bypass_l1", OPT_BOOL,
                         &gpgpu_strong_store_bypass_l1,
                         ".STRONG.GPU and .BYPASS stores skip the L1D like such "
                         "loads, as in Accel-Sim 2.0 (0 = through the L1D)",
                         "0");
  option_parser_register(opp, "-gpgpu_ldst_replay_skip_stale", OPT_BOOL,
                         &gpgpu_ldst_replay_skip_stale,
                         "a replay-queue id whose warp has nothing parked is skipped "
                         "in the same cycle (0 = it uses up the cycle's replay)",
                         "0");
  option_parser_register(opp, "-gpgpu_tlb_decouple", OPT_BOOL,
                         &gpgpu_tlb_decouple,
                         "non-blocking L1 TLB via ldst park/replay "
                         "(1 = non-blocking, default)", "1");
  option_parser_register(opp, "-gpgpu_tlb_ideal", OPT_BOOL,
                         &gpgpu_tlb_ideal,
                         "ideal TLB: instant (0-cycle) translation via functional "
                         "page-table fill; baseline to measure translation overhead", "0");
  option_parser_register(opp, "-gpgpu_l1d_vipt", OPT_BOOL,
                         &gpgpu_l1d_vipt,
                         "L1 data cache virtually indexed, physically tagged: the "
                         "L1 TLB lookup runs in parallel with the L1D access, so "
                         "an L1 TLB hit adds no latency (0 = lookup, then L1D "
                         "access). Needs the L1D bank and set inside the 4KB page "
                         "offset and every L1 TLB lookup latency <= "
                         "-gpgpu_l1_latency",
                         "0");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 32)", "32");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_option", OPT_CSTR,
                         &gpgpu_shmem_option,
                         "Option list of shared memory sizes", "0");
  option_parser_register(
      opp, "-gpgpu_unified_l1d_size", OPT_UINT32,
      &m_L1D_config.m_unified_cache_size,
      "Size of unified data cache(L1D + shared memory) in KB", "0");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_BOOL,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_UINT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_UINT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_UINT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_UINT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_UINT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_UINT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_UINT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  // The ICNT clock is also the unit of the -icnt_ring_* and -icnt_xbar_port_width parameters.
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(opp, "-icnt_inter_hop_latency_ns", OPT_UINT32,
                         &icnt_inter_hop_latency_ns,
                         "obsolete, must be 0 (inter-chiplet latency: -icnt_use_ring, "
                         "-icnt_ring_hop_latency)", "0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability",
      "32");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
  
  // MCM topology: SMs and memory partitions split into n contiguous chiplets; 1 = monolithic.
  option_parser_register(opp, "-gpgpu_n_chiplets", OPT_UINT32, &m_n_chiplets,
                       "number of chiplets in the MCM GPU",
                       "1");
  // Sizes the PMM physical-page pool that first-touch placement draws from.
  option_parser_register(opp, "-gpu_mem_size_mb", OPT_UINT32,
                        &gpu_mem_size_mb,
                        "total device memory size in MB", "4096");
  option_parser_register(opp, "-gmmu_n_ptw", OPT_UINT32,
                        &gmmu_n_ptw,
                        "number of parallel page table walkers per chiplet GMMU",
                        "16");
  option_parser_register(opp, "-gmmu_pwq_depth", OPT_UINT32,
                        &gmmu_pwq_depth,
                        "max PWQ depth per chiplet GMMU",
                        "256");
  option_parser_register(opp, "-gmmu_pwc_entries", OPT_UINT32,
                        &gmmu_pwc_entries,
                        "page walk cache entries per chiplet GMMU, shared by the three intermediate levels",
                        "128");
  // Placement map: dump first-touch placement, or load it to fix placement across configs.
  option_parser_register(opp, "-gmmu_placement_dump", OPT_CSTR,
                        &gmmu_placement_dump,
                        "write the placement map to this file at every stats dump "
                        "(empty = disabled)",
                        "");
  option_parser_register(opp, "-gmmu_placement_load", OPT_CSTR,
                        &gmmu_placement_load,
                        "pre-map every page from this placement map before the first "
                        "kernel, so the measured region takes zero faults (empty = disabled)",
                        "");
  option_parser_register(opp, "-gmmu_placement_exact", OPT_BOOL,
                        &gmmu_placement_exact,
                        "with -gmmu_placement_load: place every data page, page-table node and the "
                        "root at the PFN recorded in the (v3) map, not only on the recorded owner "
                        "chiplet. 0 = replay owners only and re-derive PFNs (pre-existing behaviour)",
                        "0");
  option_parser_register(opp, "-gmmu_local_ptw", OPT_BOOL,
                        &gmmu_local_ptw,
                        "replicate the page table on every chiplet (own root/CR3 and own "
                        "node pages) so a page walk never leaves the chiplet; data pages "
                        "are not replicated. 0 = baseline single page table",
                        "0");
  // L2 TLB configuration options
  option_parser_register(opp, "-gpgpu_l2_tlb_sets", OPT_UINT32,
                        &m_l2_tlb_sets,
                        "L2 TLB sets per chiplet (default 128)", "128");
  option_parser_register(opp, "-gpgpu_l2_tlb_assoc", OPT_UINT32,
                        &m_l2_tlb_assoc,
                        "L2 TLB associativity per chiplet (default 8)", "8");
  option_parser_register(opp, "-gpgpu_l2_tlb_set_hash", OPT_UINT32,
                        &m_l2_tlb_set_hash,
                        "L2 TLB set index (all page sizes): 0 = VPN mod sets (pre-existing), "
                        "1 = XOR-fold of the VPN in log2(sets)-bit chunks, i.e. set bit k = XOR of VPN "
                        "bits k, k+b, k+2b, ... -- the structure of the XOR set hashes reverse-engineered "
                        "on NVIDIA Pascal/Ampere L2/L3 TLBs (Nayak'21 XOR-7, NDSS'26) and confirmed on "
                        "RTX 3090. Requires power-of-two sets.",
                        "0");
  option_parser_register(opp, "-gpgpu_l2_tlb_mshr_entries", OPT_UINT32,
                        &m_l2_tlb_mshr_entries,
                        "L2 TLB MSHR unique VPN capacity per chiplet (default 128)", "128");
  option_parser_register(opp, "-gpgpu_l2_tlb_mshr_max_merge", OPT_UINT32,
                        &m_l2_tlb_mshr_max_merge,
                        "L2 TLB max merged clusters per in-flight VPN", "46");
  option_parser_register(opp, "-gpgpu_l2_tlb_lookup_latency", OPT_UINT32,
                        &m_l2_tlb_lookup_latency,
                        "L2 TLB (4KB) lookup latency in core cycles, including "
                        "the SM<->L2 TLB round trip (no separate transfer delay "
                        "is modeled)", "80");
  // L2 TLB (64KB) configuration options. 512 entries = 64 sets x 8-way.
  option_parser_register(opp, "-gpgpu_l2_tlb_64k_sets", OPT_UINT32,
                        &m_l2_tlb_64k_sets,
                        "L2 TLB (64KB) sets per chiplet (64 x 8-way = 512 entries)", "64");
  option_parser_register(opp, "-gpgpu_l2_tlb_64k_assoc", OPT_UINT32,
                        &m_l2_tlb_64k_assoc,
                        "L2 TLB (64KB) associativity per chiplet", "8");
  option_parser_register(opp, "-gpgpu_l2_tlb_64k_mshr_entries", OPT_UINT32,
                        &m_l2_tlb_64k_mshr_entries,
                        "L2 TLB (64KB) MSHR unique VPN capacity per chiplet", "128");
  option_parser_register(opp, "-gpgpu_l2_tlb_64k_mshr_max_merge", OPT_UINT32,
                        &m_l2_tlb_64k_mshr_max_merge,
                        "L2 TLB (64KB) max merged clusters per in-flight VPN", "46");
  option_parser_register(opp, "-gpgpu_l2_tlb_64k_lookup_latency", OPT_UINT32,
                        &m_l2_tlb_64k_lookup_latency,
                        "L2 TLB (64KB) lookup latency in core cycles, including "
                        "the SM<->L2 TLB round trip", "80");
  // L2 TLB (2MB) options. Unsuffixed sets/assoc/MSHR/latency are 4KB-only; set_hash, ports,
  // issue_width and starts_per_cycle apply to every page size.
  // Sizes follow the MCM-GPU reference config; MSHR equals the 4KB TLB's.
  option_parser_register(opp, "-gpgpu_l2_tlb_2m_sets", OPT_UINT32,
                        &m_l2_tlb_2m_sets,
                        "L2 TLB (2MB) sets per chiplet (32 x 8-way = 256 entries)", "32");
  option_parser_register(opp, "-gpgpu_l2_tlb_2m_assoc", OPT_UINT32,
                        &m_l2_tlb_2m_assoc,
                        "L2 TLB (2MB) associativity per chiplet", "8");
  option_parser_register(opp, "-gpgpu_l2_tlb_2m_mshr_entries", OPT_UINT32,
                        &m_l2_tlb_2m_mshr_entries,
                        "L2 TLB (2MB) MSHR unique VPN capacity per chiplet", "128");
  option_parser_register(opp, "-gpgpu_l2_tlb_2m_mshr_max_merge", OPT_UINT32,
                        &m_l2_tlb_2m_mshr_max_merge,
                        "L2 TLB (2MB) max merged clusters per in-flight VPN", "46");
  option_parser_register(opp, "-gpgpu_l2_tlb_2m_lookup_latency", OPT_UINT32,
                        &m_l2_tlb_2m_lookup_latency,
                        "L2 TLB (2MB) lookup latency in core cycles, including "
                        "the SM<->L2 TLB round trip", "80");

  option_parser_register(opp, "-gpgpu_cta_scheduling_policy", OPT_UINT32,
                        &m_cta_sched_policy,
                        "CTA-to-cluster assignment policy: 0=SEQUENTIAL (default), "
                        "1=ROUND_ROBIN_CHIPLET (chiplet-interleaved cluster order, "
                        "lead chiplet = cycle % n_chiplets), "
                        "2=STATIC_BINDING (home chiplet = cta_id % n_chiplets, "
                        "independent of timing)",
                        "0");

  option_parser_register(opp, "-gpgpu_lasp_policy", OPT_UINT32, &m_lasp_policy,
                        "LASP CTA scheduling policy (LADM MICRO'20): 0=off "
                        "(keep -gpgpu_cta_scheduling_policy behavior), "
                        "1=ROW_BINDING (grid row -> chiplet, contiguous row "
                        "blocks), 2=COL_BINDING (grid column -> chiplet), "
                        "3=CHUNK_BINDING (chunk_size consecutive CTAs "
                        "round-robin across chiplets), 4=KERNEL_PARTITION "
                        "(MCM-GPU ISCA'17 distributed CTA scheduler: the "
                        "kernel's CTAs are split into n_chiplets contiguous "
                        "ranges of ceil(#CTAs/n_chiplets); chunk computed per "
                        "kernel at launch, -gpgpu_lasp_chunk_size ignored). "
                        "Takes precedence over -gpgpu_cta_scheduling_policy "
                        "when nonzero",
                        "0");
  option_parser_register(opp, "-gpgpu_lasp_chunk_size", OPT_UINT32,
                        &m_lasp_chunk_size,
                        "CTAs per batch for -gpgpu_lasp_policy 3; align-aware "
                        "batch = page_size / datablock_size, kernel-wide "
                        "partitioning = ceil(#CTAs / n_chiplets)",
                        "1");
  option_parser_register(opp, "-gpgpu_cta_chiplet_cursor", OPT_UINT32,
                        &m_cta_chiplet_cursor,
                        "1 = distributed CTA scheduling (MCM-GPU, ISCA'17): each "
                        "chiplet issues its own CTAs from a per-chiplet cursor as "
                        "soon as it has a free slot, independent of the other "
                        "chiplets' progress (CTA->chiplet map unchanged; needs "
                        "STATIC_BINDING or a LASP policy); 0 = single sequential "
                        "cursor (legacy)",
                        "0");
  option_parser_register(opp, "-gpgpu_kernel_rebind", OPT_UINT32,
                        &m_kernel_rebind,
                        "what an SM core holding a finished kernel does when a new "
                        "kernel arrives. 0 (default): it moves to the next kernel the "
                        "trace framework creates, busy or not, without select_kernel(), "
                        "so its CTAs skip -gpgpu_kernel_launch_latency (upstream gets "
                        "the same from allocator address reuse); 1: it drops the "
                        "finished kernel and, once idle, takes one via select_kernel(), "
                        "so every kernel waits the launch latency. Both are deterministic",
                        "0");
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  note_issue_event();  // issue epoch: a new kernel can change the CTA issue scan
  unsigned kernelID = kinfo->get_uid();
  unsigned long long streamID = kinfo->get_streamID();

  kernel_time_t kernel_time = {gpu_tot_sim_cycle + gpu_sim_cycle, 0};
  if (gpu_kernel_time.find(streamID) == gpu_kernel_time.end()) {
    std::map<unsigned, kernel_time_t> new_val;
    new_val.insert(std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    gpu_kernel_time.insert(
        std::pair<unsigned long long, std::map<unsigned, kernel_time_t>>(
            streamID, new_val));
  } else {
    gpu_kernel_time.at(streamID).insert(
        std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    ////////// assume same kernel ID do not appear more than once
  }

  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }

  // LASP: set the configured CTA binding on each kernel; 0 keeps KERNEL_WIDE (no LASP).
  // The locality type comes from config, standing in for the paper's compiler pass.
  {
    unsigned pol = m_config.get_lasp_policy();
    if (pol != 0) {
      kinfo->m_lasp_config.policy =
          (pol == 1) ? lasp::ROW_BINDING
                     : (pol == 2) ? lasp::COL_BINDING : lasp::CHUNK_BINDING;
      unsigned cs = m_config.get_lasp_chunk_size();
      if (pol == 4) {
        // KERNEL_PARTITION (MCM-GPU): equal groups of contiguous CTAs, one per chiplet.
        // Done as CHUNK_BINDING, chunk = ceil(#CTAs / n_chiplets), so "% n_chiplets" is a no-op.
        unsigned n_chiplets = m_gpu_module ? m_gpu_module->get_num_chiplets()
                                           : m_config.get_num_chiplets();
        if (n_chiplets == 0) n_chiplets = 1;
        unsigned n_ctas = (unsigned)kinfo->num_blocks();
        cs = (n_ctas + n_chiplets - 1) / n_chiplets;
      }
      kinfo->m_lasp_config.chunk_size = cs ? cs : 1;
      printf("[LASP] kernel uid=%u '%s' policy=%s chunk_size=%u%s\n",
             kinfo->get_uid(), kinfo->name().c_str(),
             lasp::policy_to_string(kinfo->m_lasp_config.policy),
             kinfo->m_lasp_config.chunk_size,
             (pol == 4) ? " (KERNEL_PARTITION)" : "");
    }
  }

  // Distributed CTA scheduling: per-chiplet cursors change the issue order, not the CTA -> chiplet map.
  if (m_config.get_cta_chiplet_cursor()) {
    const bool static_bind =
        (m_config.get_cta_sched_policy() == CTA_SCHED_STATIC_BINDING);
    const bool bound =
        (kinfo->m_lasp_config.policy != lasp::KERNEL_WIDE) || static_bind;
    if (bound && m_gpu_module) {
      kinfo->pcc_init(m_gpu_module->get_num_chiplets(), static_bind);
      printf("[PCC] kernel uid=%u '%s' per-chiplet cursors on: n_chiplets=%u "
             "ctas=%u rule=%s\n",
             kinfo->get_uid(), kinfo->name().c_str(), kinfo->pcc_n(),
             (unsigned)kinfo->num_blocks(),
             (kinfo->m_lasp_config.policy != lasp::KERNEL_WIDE)
                 ? lasp::policy_to_string(kinfo->m_lasp_config.policy)
                 : "STATIC_BINDING");
    } else {
      printf("[PCC] kernel uid=%u option set but no chiplet binding active "
             "(or gpu_module not wired) -- ignored\n", kinfo->get_uid());
    }
  }

  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  // A core may still hold a deleted kernel's pointer: dereference it only if it is still running.
  // Backup check; on_kernel_deleted()/on_kernel_created() already move cores off it.
  if (kernel) {
    bool running = false;
    for (unsigned n = 0; n < m_running_kernels.size(); n++)
      if (m_running_kernels[n] == kernel) { running = true; break; }
    if (!running) return false;
  }

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

// Kernel hand-over, decided in serial code: the framework deletes a finished kernel that cores
// may still point at. Rebind 0: keep the pointer and move those cores to the next created kernel.
// Rebind 1: clear those pointers now; each idle core picks a kernel via select_kernel().
void gpgpu_sim::on_kernel_deleted(const kernel_info_t *k) {
  if (!k) return;
  if (m_config.get_kernel_rebind() == 0) {
    m_retired_kernels.push_back(k);   // pointer value only; never dereferenced
    return;
  }
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->forget_kernel(k);
}

void gpgpu_sim::on_kernel_created(kernel_info_t *nk) {
  if (m_retired_kernels.empty()) return;
  const kernel_info_t *old = m_retired_kernels.back();
  m_retired_kernels.pop_back();
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->retarget_kernel(old, nk);
}

bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency) {
      m_running_kernels[n]->m_kernel_TB_latency--;
      // Issue epoch: the launch-latency gate opening can change the CTA issue scan.
      if (!m_running_kernels[n]->m_kernel_TB_latency) note_issue_event();
    }
  }
}

kernel_info_t *gpgpu_sim::select_kernel() {
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      // Already recorded if a handed-over core issued its first CTA (note_kernel_issued()).
      if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                    launch_uid) == m_executed_kernel_uids.end()) {
        m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
        m_executed_kernel_uids.push_back(launch_uid);
        m_executed_kernel_names.push_back(m_running_kernels[idx]->name());
      }

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

// Executed-kernel list (stat header) at a kernel's first CTA. select_kernel() also adds it, but under
// -gpgpu_kernel_rebind 0 a kernel can get every CTA on handed-over cores, never through select_kernel().
void gpgpu_sim::note_kernel_issued(kernel_info_t *kernel) {
  const unsigned uid = kernel->get_uid();
  if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), uid) !=
      m_executed_kernel_uids.end())
    return;
  kernel->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
  m_executed_kernel_uids.push_back(uid);
  m_executed_kernel_names.push_back(kernel->name());
}

unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) {
    last_streamID = -1;
    return 0;
  }
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::commit_deferred_kernel_done() {
  if (m_kernel_done_pending.empty()) return;
  std::stable_sort(m_kernel_done_pending.begin(), m_kernel_done_pending.end(),
                   [](const std::pair<unsigned, kernel_info_t *> &a,
                      const std::pair<unsigned, kernel_info_t *> &b) { return a.first < b.first; });
  for (const auto &p : m_kernel_done_pending) set_kernel_done(p.second);
  m_kernel_done_pending.clear();
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  if (kernel->pcc_on()) {   // PCC: per-chiplet issue summary ([PCC] rows)
    for (unsigned c = 0; c < kernel->pcc_n(); c++)
      printf("[PCC] kernel uid=%u chiplet=%u issued=%u last_issue_cycle=%llu\n",
             kernel->get_uid(), c, kernel->pcc_issued(c),
             kernel->pcc_last_issue(c));
  }
  unsigned uid = kernel->get_uid();
  last_uid = uid;
  unsigned long long streamID = kernel->get_streamID();
  last_streamID = streamID;
  gpu_kernel_time.at(streamID).at(uid).end_cycle =
      gpu_tot_sim_cycle + gpu_sim_cycle;
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

void exec_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, m_gpu_module, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats);
}

// SST get its own simt_cluster
void sst_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new sst_simt_core_cluster(this, m_gpu_module, i, m_shader_config, m_memory_config,
                                  m_shader_stats, m_memory_stats);
  SST_gpgpu_reply_buffer.resize(m_shader_config->n_simt_clusters);
}

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
  m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
      config.g_power_simulation_enabled, config.g_power_config_name,
      config.g_power_simulation_mode, config.g_dvfs_enabled);
#endif

  m_shader_stats = new shader_core_stats(m_shader_config);
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  m_total_cta_launched = 0;
  gpu_deadlock = false;

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;
  last_streamID = -1;

  gpu_kernel_time.clear();

  // Build the chiplet topology first: icnt_configure_chiplets() below reads its owner maps.
  // The exec/SST subclass constructors call bind_global_arrays(); trace mode never reads those arrays.
  m_gpu_module = NULL;

  m_n_chiplets = m_config.get_num_chiplets();
  assert(m_n_chiplets >= 1);

  if (!m_gpu_module) {
    const unsigned n_clusters = m_shader_config->n_simt_clusters;
    const unsigned n_mem = m_memory_config->m_n_mem;
    const unsigned n_mem_parts = m_memory_config->m_n_mem_sub_partition;
    m_gpu_module = new gpu_module(this, m_n_chiplets, n_clusters, n_mem, n_mem_parts);

    m_gpu_module->init();
    m_gpu_module->set_stats_file(stdout);

    // Print the SM / memory-partition to chiplet split.
    m_gpu_module->dump_partitioning();
  }

  // Instruction and constant fetches are not translated; a real I/C cache would send them as PAs.
  if (!m_shader_config->perfect_inst_const_cache) {
    fprintf(stderr,
            "\nFATAL: -gpgpu_perfect_inst_const_cache 0 is not supported: instruction and "
            "constant fetch addresses are not translated and would alias physical frames.\n");
    fflush(stderr);
    abort();
  }

  // TODO: somehow move this logic to the sst_gpgpu_sim constructor?
  if (!m_config.is_SST_mode()) {
    // Init memory if not in SST mode
    m_memory_partition_unit =
        new memory_partition_unit *[m_memory_config->m_n_mem];
    m_memory_sub_partition =
        new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      m_memory_partition_unit[i] =
          new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
      for (unsigned p = 0;
           p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned submpid =
            i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
        m_memory_sub_partition[submpid] =
            m_memory_partition_unit[i]->get_sub_partition(p);
      }
    }

    icnt_wrapper_init();
    // Split the local crossbar per chiplet: pass the owner maps before icnt_create().
    icnt_configure_chiplets(m_n_chiplets,
                            m_gpu_module->get_cluster_to_chiplet_vec(),       // size n_simt_clusters
                            m_gpu_module->get_mem_sub_partition_to_chiplet_vec()); // size n_mem_sub_partition
    icnt_set_flit_sizes(m_memory_config->icnt_flit_size, m_memory_config->icnt_inter_flit_size);
    icnt_create(m_shader_config->n_simt_clusters,
                m_memory_config->m_n_mem_sub_partition);
  }
  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  // CTA scheduler from -gpgpu_cta_scheduling_policy (0 = upstream order).
  // Valid LASP policies are 0-4; a larger value would silently act as CHUNK binding.
  if (m_config.get_lasp_policy() > 4) {
    fprintf(stderr, "\nFATAL: -gpgpu_lasp_policy %u; valid values are 0-4.\n",
            m_config.get_lasp_policy());
    fflush(stderr);
    abort();
  }
  m_cta_scheduler = cta_scheduler_t::create(
      static_cast<cta_scheduling_policy_t>(m_config.m_cta_sched_policy));
  printf("[CTA_SCHED] policy=%s\n", m_cta_scheduler->policy_name());

  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;
}

void sst_gpgpu_sim::SST_receive_mem_reply(unsigned core_id, void *mem_req) {
  assert(core_id < m_shader_config->n_simt_clusters);
  mem_fetch *mf = (mem_fetch *)mem_req;

  (SST_gpgpu_reply_buffer[core_id]).push_back(mf);
}

mem_fetch *sst_gpgpu_sim::SST_pop_mem_reply(unsigned core_id) {
  if (SST_gpgpu_reply_buffer[core_id].size() > 0) {
    mem_fetch *temp = SST_gpgpu_reply_buffer[core_id].front();
    SST_gpgpu_reply_buffer[core_id].pop_front();
    return temp;
  } else
    return NULL;
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

bool sst_gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  if (get_more_cta_left()) return true;
  return false;
}

void gpgpu_sim::init() {
  // Parallel builds: reject configs that write across chiplets or partitions in the parallel loops.
  // abort(), not assert(), so NDEBUG cannot drop the check.
#ifdef _OPENMP
  {
    // Refuse more than 16 threads (untested; a failure would be a silently wrong result).
    // ACCELSIM_ALLOW_UNVERIFIED_THREADS overrides.
    const int omp_max = omp_get_max_threads();
    if (omp_max > 16 && getenv("ACCELSIM_ALLOW_UNVERIFIED_THREADS") == NULL) {
      fprintf(stderr,
              "\nFATAL: OMP_NUM_THREADS=%d exceeds the verified ceiling of 16.\n"
              "       Thread counts 1/2/4/8/16 reproduce the sequential result "
              "exactly;\n"
              "       higher counts have never been validated, and this class "
              "of failure is\n"
              "       silent (exit 0, wrong statistics).\n"
              "       Use OMP_NUM_THREADS<=16, or set "
              "ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 to validate a higher "
              "count.\n",
              omp_max);
      fflush(stderr);
      abort();
    }
    const char *unsafe_opt = NULL;
    // The ideal TLB maps pages on first access (shared write) unless a placement map is loaded.
    const char *placement = m_config.get_gmmu_placement_load();
    const bool placement_preloaded = placement && placement[0] != '\0';
    if (m_shader_config->gpgpu_tlb_ideal && !placement_preloaded && omp_max > 1)
      unsafe_opt = "-gpgpu_tlb_ideal 1 with more than one thread but without "
                   "-gmmu_placement_load (first access maps pages into the "
                   "GPU-wide page table; load a placement map or use "
                   "OMP_NUM_THREADS=1)";
    else if (m_memory_config->simple_dram_model)
      unsafe_opt = "simple_dram_model (not audited for partition locality)";
    if (unsafe_opt) {
      fprintf(stderr,
              "\nFATAL: this binary was built with OpenMP (parallel CORE/DRAM "
              "loops) but the\n       configuration uses %s.\n"
              "       That path writes state shared across chiplets or memory "
              "partitions and\n       is not safe under the parallel loops. "
              "Rebuild without -fopenmp, or\n       drop the option.\n",
              unsafe_opt);
      fflush(stderr);
      abort();
    }
  }
#endif

  // Defer first-touch owner inserts out of the parallel passes; they are committed once per cycle
  // in chiplet order (as 1 thread would) and read only in later cycles, so any thread count agrees.
  if (m_gpu_module && m_gpu_module->get_page_table() &&
      !m_gpu_module->get_page_table()->deferred_first_touch_enabled())
    m_gpu_module->get_page_table()->enable_deferred_first_touch(
        m_gpu_module->get_num_chiplets());

  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

// McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem,
                          m_memory_config->nbk, m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init();
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

PowerscalingCoefficients *gpgpu_sim::get_scaling_coeffs() {
  return m_gpgpusim_wrapper->get_scaling_coeffs();
}

void gpgpu_sim::print_stats(unsigned long long streamID) {
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat(streamID);

  if (g_network_mode) {
    printf(
        "----------------------------Interconnect-DETAILS----------------------"
        "----------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf(
        "----------------------------END-of-Interconnect-DETAILS---------------"
        "----------\n");
  }
  
  // Per-chiplet GMMU statistics: PTW counts, PWQ delay, local/remote steps, PWC hit rate.
  if (m_gpu_module) {
    printf("----------------------------GMMU-STATS-"
           "---------------------------------------\n");
    // [RUN_MODE]: run type from what the run actually did, not from config labels.
    // pinned is read back, so a placement file that failed to load shows as INVALID_map_not_loaded.
    {
      const char* pd = m_config.get_gmmu_placement_dump();
      const char* pl = m_config.get_gmmu_placement_load();
      const bool dumping = pd && pd[0] != '\0';
      const bool loading = pl && pl[0] != '\0';
      const bool lptw = m_gpu_module->local_ptw_enabled();
      const bool pinned = m_gpu_module->get_page_table() &&
                          m_gpu_module->get_page_table()->placement_frozen();
      const char* mode;
      if (dumping && loading && strcmp(pd, pl) == 0)
        mode = "INVALID_dump+load";  // the map would be read and overwritten
      else if (loading)  // a dump to another file is the end-state check map
        mode = !pinned ? "INVALID_map_not_loaded"
                       : (lptw ? "LOCAL_PTW" : "BASELINE");
      else if (dumping)
        mode = pinned ? "INVALID_ref_pinned"  // a reference must place by first touch
                      : "REFERENCE";          // builds the map, stats discarded
      else
        mode = "INVALID_no_placement";  // uncontrolled: neither dumped nor pinned
      printf("[RUN_MODE] mode=%s  local_ptw=%d  pinned=%d  page_shift=%u  "
             "placement=%s%s  cta_sched=%s\n",
             mode, lptw ? 1 : 0, pinned ? 1 : 0,
             m_gpu_module->uniform_page_shift(),
             dumping ? "dump:" : (loading ? "load:" : "none"),
             dumping ? pd : (loading ? pl : ""),
             m_cta_scheduler ? m_cta_scheduler->policy_name() : "?");
    }
    // Page-table footprint; with local PTW it counts one copy per chiplet.
    if (m_gpu_module->get_page_table()) {
      size_t nodes = m_gpu_module->get_page_table()->pt_node_count();
      size_t pages = m_gpu_module->get_page_table()->pt_page_count();
      printf("[PAGE_TABLE] pt_nodes=%zu  pt_pages=%zu  pt_footprint=%.3f MB  local_ptw=%d\n",
             nodes, pages, (double)(pages * GPU_PAGE_SIZE) / (1024.0 * 1024.0),
             m_gpu_module->local_ptw_enabled() ? 1 : 0);
    }
    // Total page-table changes, including build and premap, so nonzero even when pinned.
    // cycle() aborts on any change inside the parallel passes.
    printf("[PAGE_TABLE] mutations=%llu  (cumulative incl. build/premap; "
           "parallel-phase delta guarded to 0)\n",
           m_gpu_module->get_page_table()->mutation_count());
    // [REF_DEFER]: deferred first-touch commits; dedup = commits that inserted nothing,
    // new_pages = pages this run placed itself (with a loaded map, pages missing from it).
    if (m_gpu_module->get_page_table()->deferred_first_touch_enabled()) {
      const unsigned long long c = m_gpu_module->get_page_table()->deferred_ft_commits();
      const unsigned long long d = m_gpu_module->get_page_table()->deferred_ft_dedups();
      printf("[REF_DEFER] commits=%llu dedup=%llu new_pages=%llu max_batch=%llu\n",
             c, d, c - d, m_gpu_module->get_page_table()->deferred_ft_max_batch());
    }
    // Optional: write the first-touch placement map for a later measurement run.
    {
      const char* pd = m_config.get_gmmu_placement_dump();
      if (pd && pd[0] != '\0' && m_gpu_module->get_page_table())
        m_gpu_module->get_page_table()->dump_placement(pd);
    }
    for (unsigned c = 0; c < m_gpu_module->get_num_chiplets(); c++) {
      m_gpu_module->get_gmmu(c)->print_stats(stdout);
    }
    // With one page size, skip the two unused TLBs; uniform_page_shift() is 0 for mixed policies.
    const unsigned ups = m_gpu_module->uniform_page_shift();
    auto want_kind = [ups](unsigned shift) { return ups == 0 || ups == shift; };
    // L1 TLB statistics per SM, one TLB per page size
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        for (unsigned j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
            shader_core_ctx* core = m_cluster[i]->get_core(j);
            if (!core) continue;
            if (core->get_l1_tlb()     && want_kind(GPU_PAGE_SHIFT)) core->get_l1_tlb()->print_stats(stdout);
            if (core->get_l1_tlb_64k() && want_kind(GPU_64K_SHIFT))  core->get_l1_tlb_64k()->print_stats(stdout);
            if (core->get_l1_tlb_2m()  && want_kind(GPU_HUGE_SHIFT)) core->get_l1_tlb_2m()->print_stats(stdout);
            core->xlat_print_stats(stdout);   // [LDST_XLATLAT] row, per SM
        }
    }
    xlatsum_gpu_print(stdout);  // GPU-wide [XLATSUM] summary rows
    // L2 TLB statistics per chiplet, one TLB per page size
    for (unsigned c = 0; c < m_gpu_module->get_num_chiplets(); c++) {
      if (want_kind(GPU_PAGE_SHIFT)) m_gpu_module->get_chiplet(c)->get_l2_tlb_4k()->print_stats(stdout);
      if (want_kind(GPU_64K_SHIFT))  m_gpu_module->get_chiplet(c)->get_l2_tlb_64k()->print_stats(stdout);
      if (want_kind(GPU_HUGE_SHIFT)) m_gpu_module->get_chiplet(c)->get_l2_tlb_2m()->print_stats(stdout);
    }
    // Diagnostic rows ([ACCEPT], [POPGATE], [NORMREP], [NORMREP-MTX]): only under ACCELSIM_DIAG.
    if (g_accelsim_diag) {
    // [ACCEPT]: per-chiplet share of L2 visits with a full input, and DRAM utilization.
    {
      unsigned nc = m_gpu_module->get_num_chiplets(); if (nc > 8) nc = 8;
      // aggregate per-chiplet DRAM bwutil over that chiplet's memory channels
      double dram_util_sum[8] = {0}; unsigned dram_ch_cnt[8] = {0};
      for (unsigned p = 0; p < m_memory_config->m_n_mem; p++) {
        unsigned ch = m_gpu_module->mem_sub_partition_to_chiplet(
            p * m_memory_config->m_n_sub_partition_per_memory_channel);
        if (ch < 8) { dram_util_sum[ch] += m_memory_partition_unit[p]->get_dram_bwutil();
                      dram_ch_cnt[ch]++; }
      }
      for (unsigned c = 0; c < nc; c++) {
        unsigned long long tot = g_memfull_total[c] ? g_memfull_total[c] : 1;
        printf("[ACCEPT] chiplet=%u mem_full=%llu/%llu (%.1f%%)  DRAM_util=%.1f%%\n",
               c, g_memfull_stall[c], g_memfull_total[c],
               100.0 * g_memfull_stall[c] / tot,
               dram_ch_cnt[c] ? 100.0 * dram_util_sum[c] / dram_ch_cnt[c] : 0.0);
      }
    }
    // [POPGATE]: icnt-pop gating of each chiplet's first cluster only (a lower bound).
    for (unsigned c = 0; c < m_gpu_module->get_num_chiplets(); c++) {
      unsigned repr = m_gpu_module->get_chiplet(c)->get_cluster_global_begin();
      unsigned long long icnt_cycles = 0, gated = 0;
      m_cluster[repr]->get_popgate_stats(icnt_cycles, gated);
      printf("[POPGATE] repr_cluster=%u chiplet=%u icnt_cycles=%llu gated=%llu (%.1f%%) "
             "[repr-cluster share only; a lower bound with round-robin PTW endpoints]\n",
             repr, c, icnt_cycles, gated,
             icnt_cycles ? 100.0 * gated / icnt_cycles : 0.0);
    }
    // [NORMREP]: non-PTW reply xbar latency by source memory chiplet (compare with PTW [FUNNEL]).
    {
      extern unsigned long long g_normrep_transit_sum[8], g_normrep_lasthop_sum[8],
          g_normrep_cnt[8];
      for (unsigned c = 0; c < m_gpu_module->get_num_chiplets() && c < 8; c++) {
        unsigned long long n = g_normrep_cnt[c] ? g_normrep_cnt[c] : 1;
        printf("[NORMREP] src_chiplet=%u normal-reply: transit_avg=%.1f lasthop_avg=%.1f "
               "[n=%llu]\n", c, (double)g_normrep_transit_sum[c] / n,
               (double)g_normrep_lasthop_sum[c] / n, g_normrep_cnt[c]);
      }
      // [NORMREP-MTX]: normal-reply transit_avg matrix [src mem chiplet][dst cluster chiplet].
      extern unsigned long long g_normrep_mtx_transit[8][8], g_normrep_mtx_cnt[8][8];
      unsigned nc = m_gpu_module->get_num_chiplets(); if (nc > 8) nc = 8;
      printf("[NORMREP-MTX] transit_avg  rows=src_mem_chiplet  cols=dst_cluster_chiplet\n");
      for (unsigned s = 0; s < nc; s++) {
        printf("[NORMREP-MTX] src%u:", s);
        for (unsigned d = 0; d < nc; d++) {
          unsigned long long m = g_normrep_mtx_cnt[s][d];
          printf("  d%u=%.0f(n=%llu)", d,
                 m ? (double)g_normrep_mtx_transit[s][d] / m : 0.0, m);
        }
        printf("\n");
      }
    }
    }  // g_accelsim_diag: ACCEPT/POPGATE/NORMREP rows
    printf("----------------------------END-of-GMMU-STATS-"
           "----------------------------------\n");
  }

  // PHASE_TIMING totals so far, to stderr, at every stats dump (a crashed run still keeps one).
  if (g_phase_timing_on) {
    const unsigned long long total_ns =
        g_phase_timers.dram_ns + g_phase_timers.l2gmmu_ns + g_phase_timers.icnt_ns +
        g_phase_timers.phase_a_ns + g_phase_timers.phase_b_ns + g_phase_timers.phase_c_ns;
    const double pct = total_ns ? 100.0 / (double)total_ns : 0.0;
    fprintf(stderr,
        "[PHASE_TIMING] threads=%d total=%.3fs "
        "dram=%.1f%%(n=%llu) l2gmmu=%.1f%%(n=%llu) icnt=%.1f%%(n=%llu) "
        "phase_a=%.1f%%(n=%llu) phase_b=%.1f%%(n=%llu) phase_c=%.1f%%(n=%llu)\n",
#ifdef _OPENMP
        omp_get_max_threads(),
#else
        1,
#endif
        (double)total_ns / 1.0e9,
        g_phase_timers.dram_ns * pct, g_phase_timers.dram_calls,
        g_phase_timers.l2gmmu_ns * pct, g_phase_timers.l2gmmu_calls,
        g_phase_timers.icnt_ns * pct, g_phase_timers.icnt_calls,
        g_phase_timers.phase_a_ns * pct, g_phase_timers.phase_a_calls,
        g_phase_timers.phase_b_ns * pct, g_phase_timers.phase_b_calls,
        g_phase_timers.phase_c_ns * pct, g_phase_timers.phase_c_calls);
    if (g_phase_timers.phase_b_calls) {  // [PHASE_TIMING_B]: phase B per chiplet group
      fprintf(stderr, "[PHASE_TIMING_B] chip_s=");
      for (unsigned c = 0; c < 8; c++)
        if (g_phase_timers.phase_b_chip_ns[c])
          fprintf(stderr, "%s%u:%.3f", c ? "," : "", c,
                  (double)g_phase_timers.phase_b_chip_ns[c] / 1.0e9);
      fprintf(stderr, " runs_by_thread=");
      for (unsigned c = 0; c < 8; c++) {
        if (!g_phase_timers.phase_b_chip_ns[c]) continue;
        fprintf(stderr, "%s%u:{", c ? " " : "", c);
        bool first = true;
        for (unsigned t = 0; t < 16; t++)
          if (g_phase_timers.phase_b_chip_thread[c][t]) {
            fprintf(stderr, "%st%u=%llu", first ? "" : ",", t,
                    g_phase_timers.phase_b_chip_thread[c][t]);
            first = false;
          }
        fprintf(stderr, "}");
      }
      fprintf(stderr, "\n");
      // per-cycle maximum over groups and log2(us) histograms
      fprintf(stderr, "[PHASE_TIMING_B2] cyclemax=%.3fs argmax=",
              (double)g_phase_timers.phase_b_cyclemax_ns / 1.0e9);
      for (unsigned c = 0; c < 8; c++)
        if (g_phase_timers.phase_b_argmax_cnt[c])
          fprintf(stderr, "%s%u:%llu", c ? "," : "", c, g_phase_timers.phase_b_argmax_cnt[c]);
      fprintf(stderr, " max_hist_log2us=");
      for (unsigned b = 0; b < 24; b++)
        if (g_phase_timers.phase_b_max_hist[b])
          fprintf(stderr, "%s%u:%llu", b ? "," : "", b, g_phase_timers.phase_b_max_hist[b]);
      fprintf(stderr, " body_hist_log2us=");
      for (unsigned b = 0; b < 24; b++)
        if (g_phase_timers.phase_b_body_hist[b])
          fprintf(stderr, "%s%u:%llu", b ? "," : "", b, g_phase_timers.phase_b_body_hist[b]);
      fprintf(stderr, "\n");
    }
    // Coverage: "uncov" is cycle() time outside every bucket above.
    const unsigned long long bucket_sum = total_ns +
        g_phase_timers.cluster_icnt_ns + g_phase_timers.tail_ns;
    fprintf(stderr,
        "[PHASE_TIMING_COVER] cycle=%.3fs cluster_icnt=%.3fs tail=%.3fs "
        "uncov_in_cycle=%.3fs\n",
        g_phase_timers.cycle_ns / 1.0e9,
        g_phase_timers.cluster_icnt_ns / 1.0e9,
        g_phase_timers.tail_ns / 1.0e9,
        (g_phase_timers.cycle_ns > bucket_sum
             ? (g_phase_timers.cycle_ns - bucket_sum) / 1.0e9 : 0.0));
    fprintf(stderr, "[PHASE_TIMING_TAIL] issue=%.3fs occ=%.3fs rest=%.3fs\n",
        g_phase_timers.tail_issue_ns / 1.0e9, g_phase_timers.tail_occ_ns / 1.0e9,
        (g_phase_timers.tail_ns - g_phase_timers.tail_issue_ns -
         g_phase_timers.tail_occ_ns) / 1.0e9);
    // ICNT_POP: the three parts of the icnt phase.
    // ICNT_SPLIT: icnt_transfer() as parallel Advance() vs. serial handoff, in % of icnt_transfer().
    fprintf(stderr,
        "[PHASE_TIMING_ICNT_POP] mem2icnt_pop=%.3fs(n=%llu) xbar_advance=%.3fs "
        "handoff=%.3fs  sum=%.3fs  icnt_phase=%.3fs\n",
        g_icnt_pop_ns / 1.0e9, g_icnt_pop_calls, g_icnt_local_ns / 1.0e9,
        g_icnt_handoff_ns / 1.0e9,
        (g_icnt_pop_ns + g_icnt_local_ns + g_icnt_handoff_ns) / 1.0e9,
        g_phase_timers.icnt_ns / 1.0e9);
    const unsigned long long icnt_sub_ns = g_icnt_local_ns + g_icnt_handoff_ns;
    const double icnt_pct = icnt_sub_ns ? 100.0 / (double)icnt_sub_ns : 0.0;
    fprintf(stderr,
        "[PHASE_TIMING_ICNT_SPLIT] local_advance=%.1f%%(%.3fs,n=%llu) "
        "handoff=%.1f%%(%.3fs,n=%llu)\n",
        g_icnt_local_ns * icnt_pct, (double)g_icnt_local_ns / 1.0e9, g_icnt_local_calls,
        g_icnt_handoff_ns * icnt_pct, (double)g_icnt_handoff_ns / 1.0e9, g_icnt_handoff_calls);
    fflush(stderr);
  }
}

void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    // Last writeback over all SMs (lowest SM id on a tie), from per-SM cycles.
    unsigned long long last_wb = 0;
    unsigned last_wb_sid = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      for (unsigned j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
        const shader_core_ctx *core = m_cluster[i]->get_core(j);
        if (core->last_writeback_cycle() > last_wb) {
          last_wb = core->last_writeback_cycle();
          last_wb_sid = core->get_sid();
        }
      }
    const unsigned long long now = gpu_tot_sim_cycle + gpu_sim_cycle;
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ cycle %llu (%llu cycles ago)\n",
        last_wb_sid, last_wb, now >= last_wb ? now - last_wb : 0ULL);
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    // Deadlock dump (XLATDBG_*): each stuck request with its status and the cycle it last changed.
    {
      const unsigned subs = m_memory_config->m_n_sub_partition_per_memory_channel;
      printf("XLATDBG_DEADLOCK: ===== stuck memory requests =====\n");
      for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
        if (!m_memory_partition_unit[i]->busy()) continue;
        const unsigned chip = m_gpu_module ? m_gpu_module->chiplet_of_mem(i) : 0;
        for (unsigned s = 0; s < subs; s++) {
          printf("XLATDBG_DEADLOCK: mem_part=%u (chiplet %u) sub=%u\n", i, chip, s);
          m_memory_partition_unit[i]->get_sub_partition(s)->print(stdout);
        }
      }
      printf("XLATDBG_DEADLOCK: ===== address-translation waiters =====\n");
      for (unsigned c = 0; c < m_shader_config->n_simt_clusters; c++)
        m_cluster[c]->xlatdbg_dump_waiters(stdout);
      if (m_gpu_module)
        for (unsigned c = 0; c < m_gpu_module->get_num_chiplets(); c++) {
          m_gpu_module->get_gmmu(c)->xlatdbg_dump_pending(stdout);
          const chiplet *ch = m_gpu_module->get_chiplet(c);
          const gpu_l2_tlb *l2[3] = {ch->get_l2_tlb_4k(), ch->get_l2_tlb_64k(), ch->get_l2_tlb_2m()};
          const char *tag[3] = {"4k", "64k", "2m"};
          for (unsigned k = 0; k < 3; k++)
            if (l2[k]) l2[k]->xlatdbg_dump(stdout, c, tag[k]);
        }
      printf("XLATDBG_DEADLOCK: ===== end =====\n");
      fflush(stdout);
    }
    if (icnt_busy()) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

std::string gpgpu_sim::executed_kernel_name() {
  std::stringstream statout;
  if (m_executed_kernel_names.size() == 1)
    statout << m_executed_kernel_names[0];
  else {
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " ";
    }
  }
  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}

void gpgpu_sim::gpu_print_stat(unsigned long long streamID) {
  FILE *statfout = stdout;

  std::string kernel_info_str = executed_kernel_info_string();
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("kernel_stream_id = %llu\n", streamID);

  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  printf("gpu_stall_dramfull = %llu\n", gpu_stall_dramfull);
  printf("gpu_stall_icnt2sh    = %llu\n", gpu_stall_icnt2sh);

  // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  // printf("partiton_reqs_in_parallel_total    = %lld\n",
  // partiton_reqs_in_parallel_total );
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  // printf("partiton_reqs_in_parallel_util = %lld\n",
  // partiton_reqs_in_parallel_util);
  // printf("partiton_reqs_in_parallel_util_total    = %lld\n",
  // partiton_reqs_in_parallel_util_total ); printf("gpu_sim_cycle_parition_util
  // = %lld\n", gpu_sim_cycle_parition_util);
  // printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
  // gpu_tot_sim_cycle_parition_util );
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  // printf("partiton_replys_in_parallel = %lld\n",
  // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
  // %lld\n", partiton_replys_in_parallel_total );
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.core_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.core_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, streamID,
                               "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout, streamID,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);

  m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    if (m_config.g_power_simulation_mode > 0) {
      // if(!m_config.g_aggregate_power_stats)
      mcpat_reset_perf_count(m_gpgpusim_wrapper);
      calculate_hw_mcpat(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                         m_power_stats, m_config.gpu_stat_sample_freq,
                         gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                         gpu_sim_insn, m_config.g_power_simulation_mode,
                         m_config.g_dvfs_enabled, m_config.g_hw_perf_file_name,
                         m_config.g_hw_perf_bench_name, executed_kernel_name(),
                         m_config.accelwattch_hybrid_configuration,
                         m_config.g_aggregate_power_stats);
    }
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);
    // if(!m_config.g_aggregate_power_stats)
    mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif

  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    flush_l2loop_idle();  // add the port samples of skipped visits first
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, streamID, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, streamID,
                                "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  // Print MCM access statistics
  if (m_gpu_module) {
    printf("\n========= MCM Access Statistics (data packets, request+reply, PTW excluded) =========\n");
    printf("mcm_remote_accesses=%llu\n", m_gpu_module->get_remote_access_count());
    printf("mcm_local_accesses=%llu\n", m_gpu_module->get_local_access_count());
    unsigned long long total_accesses = m_gpu_module->get_remote_access_count() +
                                         m_gpu_module->get_local_access_count();
    if (total_accesses > 0) {
      double remote_ratio = (double)m_gpu_module->get_remote_access_count() / total_accesses * 100.0;
      printf("mcm_remote_ratio=%.2f%%\n", remote_ratio);
    }
    printf("=========================================\n");
  }

  // PMM stats: physical frames still free in each chiplet's pool.
  if (m_gpu_module) {
    printf("\n========= PMM Statistics =========\n");
    for (unsigned c = 0; c < m_gpu_module->get_num_chiplets(); c++)
      printf("pmm_chiplet%u_free_pages=%zu\n",
             c, m_gpu_module->get_chiplet(c)->free_page_count());
    printf("===================================\n");
  }

  time_vector_print();
  fflush(stdout);

  clear_executed_kernel_info();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      {  // shared statistic, updated by every SM in parallel: atomic
      #pragma omp atomic
        m_stats->gpgpu_n_shmem_insn += active_count;
      }
      break;
    case sstarr_space:
      {  // shared statistic, updated by every SM in parallel: atomic
      #pragma omp atomic
        m_stats->gpgpu_n_sstarr_insn += active_count;
      }
      break;
    case const_space:
      {  // shared statistic, updated by every SM in parallel: atomic
      #pragma omp atomic
        m_stats->gpgpu_n_const_insn += active_count;
      }
      break;
    case param_space_kernel:
    case param_space_local:
      {  // shared statistic, updated by every SM in parallel: atomic
      #pragma omp atomic
        m_stats->gpgpu_n_param_insn += active_count;
      }
      break;
    case tex_space:
      {  // shared statistic, updated by every SM in parallel: atomic
      #pragma omp atomic
        m_stats->gpgpu_n_tex_insn += active_count;
      }
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        {  // shared statistic, updated by every SM in parallel: atomic
        #pragma omp atomic
          m_stats->gpgpu_n_store_insn += active_count;
        }
      else
        {  // shared statistic, updated by every SM in parallel: atomic
        #pragma omp atomic
          m_stats->gpgpu_n_load_insn += active_count;
        }
      break;
    default:
      abort();
  }
}
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    // One kernel per SM: the distributed and strict-affinity paths may offer another kernel's CTA.
    if (m_n_active_cta > 0 && m_cta_kernel != &kernel) return false;
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  const unsigned smem_per_cta =  // -gpgpu_shmem_per_cta_reserve, as in max_cta()
      kernel_info->smem > 0 ? kernel_info->smem + m_config->gpgpu_shmem_per_cta_reserve : 0;
  if (m_occupied_shmem + smem_per_cta > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += smem_per_cta;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sid);
  }

  return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    const unsigned smem_per_cta =
        kernel_info->smem > 0 ? kernel_info->smem + m_config->gpgpu_shmem_per_cta_reserve : 0;
    assert(m_occupied_shmem >= smem_per_cta);
    m_occupied_shmem -= smem_per_cta;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();
  m_cta_kernel = &kernel;
  m_gpu->note_kernel_issued(&kernel);

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;
  if (accelsim_log_cta()) {
    // x,y,z from the linear id: the dim3 cursor already points at the next CTA.
    dim3 g = kernel.get_grid_dim();
    dim3 xyz;
    xyz.x = ctaid % g.x; xyz.y = (ctaid / g.x) % g.y; xyz.z = ctaid / (g.x * g.y);
    // Use the owner map: integer division is wrong when clusters do not split evenly.
    unsigned chiplet = m_gpu->get_gpu_module()
                           ? m_gpu->get_gpu_module()->cluster_to_chiplet(m_tpc)
                           : 0;
    printf("[CTA_ISSUE] kernel=%u cta=%u xyz=(%u,%u,%u) sm=%u cluster=%u chiplet=%u hw_cta=%u cycle=%llu\n",
           kernel.get_uid(), ctaid, xyz.x, xyz.y, xyz.z, m_sid, m_tpc, chiplet,
           free_cta_hw_id,
           (unsigned long long)(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle));
  }

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

// CTA scheduler implementations

cta_scheduler_t* cta_scheduler_t::create(cta_scheduling_policy_t policy) {
  switch (policy) {
    case CTA_SCHED_SEQUENTIAL:
      return new SequentialScheduler();
    case CTA_SCHED_ROUND_ROBIN_CHIPLET:
      return new RoundRobinChipletScheduler();
    case CTA_SCHED_STATIC_BINDING:
      return new StaticBindingScheduler();
    default:
      fprintf(stderr, "\nFATAL: unknown -gpgpu_cta_scheduling_policy %d (valid: 0-2)\n", (int)policy);
      fflush(stderr);
      abort();
  }
}

// Upstream order: clusters from one past last_issued, wrapping around.
std::vector<unsigned> SequentialScheduler::cluster_order(
    unsigned n_clusters, unsigned /*n_chiplets*/, unsigned last_issued,
    unsigned long long /*cycle*/) {
  std::vector<unsigned> order;
  order.reserve(n_clusters);
  for (unsigned i = 0; i < n_clusters; i++)
    order.push_back((i + last_issued + 1) % n_clusters);
  return order;
}

// Chiplet-interleaved visit order. The lead chiplet comes from the cycle, not the call
// count, so skipped scans do not change it.
std::vector<unsigned> RoundRobinChipletScheduler::cluster_order(
    unsigned n_clusters, unsigned n_chiplets, unsigned /*last_issued*/,
    unsigned long long cycle) {
  assert(n_chiplets > 0 && n_clusters % n_chiplets == 0);
  unsigned clusters_per_chiplet = n_clusters / n_chiplets;
  const unsigned start = (unsigned)(cycle % n_chiplets);

  std::vector<unsigned> order;
  order.reserve(n_clusters);

  for (unsigned slot = 0; slot < clusters_per_chiplet; slot++) {
    for (unsigned k = 0; k < n_chiplets; k++) {
      unsigned chip = (start + k) % n_chiplets;
      order.push_back(chip * clusters_per_chiplet + slot);
    }
  }

  return order;
}

// Policy 0 (SequentialScheduler) keeps the upstream visit order.
void gpgpu_sim::issue_block2core() {
  // No CTAs left: the scan below would change nothing, so skip it.
  if (!get_more_cta_left()) return;
  // Skip the scan if nothing changed since a scan that issued nothing (CTA exit, kernel launch
  // and TB_latency reaching 0 bump the epoch). Exact: a repeat would issue nothing either.
  if (m_issue_seen_epoch == m_issue_event_epoch) return;
  unsigned n_clusters = m_shader_config->n_simt_clusters;
  unsigned n_chiplets = m_gpu_module ? m_gpu_module->get_num_chiplets() : 1;

  // Cluster visit order comes from the configured CTA scheduling policy.
  std::vector<unsigned> order =
      m_cta_scheduler->cluster_order(n_clusters, n_chiplets, m_last_cluster_issue,
                                     gpu_sim_cycle + gpu_tot_sim_cycle);

  unsigned issued_total = 0;
  for (unsigned idx : order) {
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
      issued_total += num;
    }
  }
  // A scan that issued nothing sleeps until the next issue event.
  if (issued_total == 0) m_issue_seen_epoch = m_issue_event_epoch;
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline

// Wires the busy flags on the first cycle(), once clusters and ICNT routers exist.
// All flags start at 1, so work queued before wiring is still visited.
void gpgpu_sim::wire_icnt_mirrors() {
  m_icnt_mirrors_wired = true;
  const unsigned n_sub = m_memory_config->m_n_mem_sub_partition;
  m_l2icnt_nonempty.assign(n_sub, 1);
  for (unsigned i = 0; i < n_sub; i++)
    m_memory_sub_partition[i]->set_l2icnt_mirror(&m_l2icnt_nonempty[i]);

  const unsigned n_cl = m_shader_config->n_simt_clusters;
  m_cluster_icnt_busy.assign(n_cl, 1);
  // No skipping under SST (own reply FIFO) or intersim2 (no output-buffer hook).
  m_cluster_icnt_mirror_ok = icnt_out_mirror_supported() && !m_memory_config->SST_mode;
  if (m_cluster_icnt_mirror_ok) {
    for (unsigned i = 0; i < n_cl; i++) {
      assert(m_cluster[i]->get_cluster_id() == i);   // same device number as icnt_pop(m_cluster_id)
      m_cluster[i]->set_icnt_busy_mirror(&m_cluster_icnt_busy[i]);
      icnt_set_out_mirror(i, &m_cluster_icnt_busy[i]);
    }
  }

  // L2 loop skipping needs the local xbar and no per-visit work (power, visualizer, SST).
  m_l2loop_busy.assign(n_sub, 1);
  m_l2loop_last.assign(n_sub, 0);
  m_l2loop_mirror_ok = icnt_out_mirror_supported() && !m_memory_config->SST_mode &&
                       !m_config.g_power_simulation_enabled && !m_config.g_visualizer_enabled;
  if (m_l2loop_mirror_ok) {
    for (unsigned i = 0; i < n_sub; i++) {
      m_memory_sub_partition[i]->set_l2loop_mirror(&m_l2loop_busy[i]);
      icnt_set_out_mirror(m_shader_config->mem2device(i), &m_l2loop_busy[i]);
    }
  }
}

// Adds the idle port samples of skipped L2 visits; call before reading L2 stats.
void gpgpu_sim::flush_l2loop_idle() {
  if (!m_l2loop_mirror_ok) return;
  for (unsigned i = 0; i < m_l2loop_last.size(); i++) {
    const unsigned long long n = m_l2loop_tick - m_l2loop_last[i];
    if (n) {
      m_memory_sub_partition[i]->add_idle_l2loop_visits(n);
      m_l2loop_last[i] = m_l2loop_tick;
    }
  }
}

void gpgpu_sim::cycle() {
  // Phase timing (ACCELSIM_PHASE_TIMING): wall time of the whole cycle().
  const unsigned long long cycle_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
  unsigned long long tail_wall_start = 0;  // set in the CORE block, read at the end of cycle()
  int clock_mask = next_clock_domain();
  // First call only: wire the busy flags used to skip idle clusters and sub-partitions.
  if (!m_icnt_mirrors_wired) wire_icnt_mirrors();

  if (clock_mask & ICNT) {
    const unsigned long long cluster_icnt_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
    // ICNT -> cluster replies, on the ICNT clock. Skip clusters whose response FIFO and ICNT
    // output are empty (icnt_cycle() would do nothing), except under ACCELSIM_DIAG.
    const bool skip_idle_clusters = m_cluster_icnt_mirror_ok && !g_accelsim_diag;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (skip_idle_clusters && !m_cluster_icnt_busy[i]) {
#ifdef H33_MIRROR_CHECK   // build with -DH33_MIRROR_CHECK to assert the skip is a no-op
        assert(m_cluster[i]->icnt_response_empty() && icnt_out_empty(i));
#endif
        continue;
      }
      m_cluster[i]->icnt_cycle();
      if (m_cluster_icnt_mirror_ok)
        m_cluster_icnt_busy[i] =
            (!m_cluster[i]->icnt_response_empty() || !icnt_out_empty(i)) ? 1 : 0;
    }
    if (g_phase_timing_on)
      g_phase_timers.cluster_icnt_ns += phase_now_ns() - cluster_icnt_wall_start;
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (clock_mask & L2) {
    const unsigned long long icnt_pop_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
    // Move at most one reply per sub-partition from L2 to ICNT, on the L2 clock.
    // Empty queues are skipped: top() would return NULL and pop() would do nothing.
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      if (!m_l2icnt_nonempty[i]) {
#ifdef H33_MIRROR_CHECK
        assert(m_memory_sub_partition[i]->l2icnt_queue_empty());
#endif
        continue;
      }
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        // Packet size in bytes: a write ack carries only the control part.
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          m_gpu_module->icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;  // input buffer full: retry next L2 cycle
        }
      } else {
        // top() == NULL: the head slot is empty; pop() advances it.
        m_memory_sub_partition[i]->pop();
      }
      m_l2icnt_nonempty[i] = m_memory_sub_partition[i]->l2icnt_queue_empty() ? 0 : 1;
    }
    if (g_phase_timing_on) {
      const unsigned long long icnt_pop_wall_end = phase_now_ns();
      // icnt_ns also includes icnt_transfer() below (which counts the call); g_icnt_pop_ns is this loop only.
      g_phase_timers.icnt_ns += icnt_pop_wall_end - icnt_pop_wall_start;
      g_icnt_pop_ns += icnt_pop_wall_end - icnt_pop_wall_start;
      g_icnt_pop_calls++;
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    const unsigned long long dram_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
    // Partitions touch only their own state (shared memory_stats_t is atomic), so run in parallel.
#pragma omp parallel for schedule(runtime)
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else  // DRAM->L2 return, DRAM timing model, L2->DRAM issue.
        m_memory_partition_unit[i]->dram_cycle();
      // Copy this channel's DRAM counters into the power-model stats.
      m_memory_partition_unit[i]->set_dram_power_stats(
          m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
    }
    if (g_phase_timing_on) {
      g_phase_timers.dram_ns += phase_now_ns() - dram_wall_start;
      g_phase_timers.dram_calls++;
    }
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (clock_mask & L2) {
    const unsigned long long l2_gmmu_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
    // Serial: each call is too small to pay for threads. GMMU is per chiplet, on the L2 clock.
    for (unsigned c = 0; c < m_n_chiplets; c++) {
      m_gpu_module->get_gmmu(c)->cycle(
          gpu_sim_cycle + gpu_tot_sim_cycle);
    }

    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    // Skip idle sub-partitions: their only work, an idle L2-port sample, is added later.
    // No skipping under ACCELSIM_DIAG, which counts every visit.
    const unsigned long long l2tick = ++m_l2loop_tick;
    const bool skip_idle_l2 = m_l2loop_mirror_ok && !g_accelsim_diag;
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      if (skip_idle_l2 && !m_l2loop_busy[i]) {
#ifdef H33_MIRROR_CHECK
        assert(m_memory_sub_partition[i]->l2loop_idle(SECTOR_CHUNCK_SIZE) &&
               icnt_out_empty(m_shader_config->mem2device(i)));
#endif
        continue;
      }
      if (m_l2loop_mirror_ok) {
        const unsigned long long skipped = l2tick - m_l2loop_last[i] - 1;
        if (skipped) m_memory_sub_partition[i]->add_idle_l2loop_visits(skipped);
        m_l2loop_last[i] = l2tick;
      }
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      // Diagnostics (ACCELSIM_DIAG): per-chiplet count of visits with a full L2 input queue.
      if (g_accelsim_diag) {
        unsigned mch = m_gpu_module ? m_gpu_module->mem_sub_partition_to_chiplet(i) : 0;
        if (mch < 8) {
          g_memfull_total[mch]++;
          if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) g_memfull_stall[mch]++;
        }
      }
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        if (mf) partiton_reqs_in_parallel_per_cycle++;
      }
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      if (m_config.g_power_simulation_enabled) {
        m_memory_sub_partition[i]->accumulate_L2cache_stats(
            m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
      }
      if (m_l2loop_mirror_ok)  // busy unless this visit left nothing to do
        m_l2loop_busy[i] = (m_memory_sub_partition[i]->l2loop_idle(SECTOR_CHUNCK_SIZE) &&
                            icnt_out_empty(m_shader_config->mem2device(i))) ? 0 : 1;
    }
    if (g_phase_timing_on) {
      g_phase_timers.l2gmmu_ns += phase_now_ns() - l2_gmmu_wall_start;
      g_phase_timers.l2gmmu_calls++;
    }
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    const unsigned long long icnt_transfer_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
    // Cycle stamp for the ACCELSIM_DIAG latency counters.
    if (g_accelsim_diag) g_icnt_now = gpu_sim_cycle + gpu_tot_sim_cycle;
    icnt_transfer();
    if (g_phase_timing_on) {
      g_phase_timers.icnt_ns += phase_now_ns() - icnt_transfer_wall_start;
      g_phase_timers.icnt_calls++;
    }
  }

  if (clock_mask & CORE) {
    // Group clusters by chiplet, once (lazily: gpu_module is not wired at construction).
    if (m_chiplet_cluster_order.empty()) {
      const unsigned n_chiplets =
          m_gpu_module ? m_gpu_module->get_num_chiplets() : 1u;
      m_chiplet_cluster_order.assign(n_chiplets, std::vector<unsigned>());
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        const unsigned owner =
            m_gpu_module ? m_gpu_module->cluster_to_chiplet(i) : 0u;
        assert(owner < n_chiplets);
        m_chiplet_cluster_order[owner].push_back(i);
      }
      // Every cluster must appear exactly once, or SMs would be silently dropped.
      unsigned total = 0;
      for (unsigned c = 0; c < n_chiplets; c++)
        total += m_chiplet_cluster_order[c].size();
      assert(total == m_shader_config->n_simt_clusters);
    }
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    // Hoisted out of the loop: kernel slots change only after the passes and CTA issue comes later.
    const bool more_cta_left = get_more_cta_left();
    unsigned long long active_sms_this_cycle = 0;
    // The GPU-wide page table must not change during the parallel passes (its maps are not
    // thread-safe): sample its mutation counter here and abort after pass C if it moved.
#ifdef _OPENMP
    // Only with more than one thread; a 1-thread run may fill the table (that builds the map).
    const bool pt_guard_active = (omp_get_max_threads() > 1);
    const unsigned long long pt_mut_before =
        (pt_guard_active && m_gpu_module)
            ? m_gpu_module->get_page_table()->mutation_count() : 0ull;
#endif
    const unsigned n_chiplet_groups = m_chiplet_cluster_order.size();

    // CORE = three passes: A (writeback, non-ldst units, ldst cycle_a; per SM), B (ldst cycle_b:
    // chiplet L2 TLB / GMMU; one chiplet per thread), C (ldst cycle_c, issue, fetch; per SM).
    // Pass A's run/skip decision is reused by B and C, as upstream tests once per cycle.
    if (m_cluster_ran.size() != m_shader_config->n_simt_clusters)
      m_cluster_ran.assign(m_shader_config->n_simt_clusters, 0);

    // ACCELSIM_SERIAL_PHASE=A|B|C runs that pass on one thread (to locate a race). Read once.
    static const char *serial_phase_env = getenv("ACCELSIM_SERIAL_PHASE");
    static const bool par_a = !(serial_phase_env && *serial_phase_env == 'A');
    static const bool par_b = !(serial_phase_env && *serial_phase_env == 'B');
    static const bool par_c = !(serial_phase_env && *serial_phase_env == 'C');

    // ACCELSIM_SERIAL_SUBSYS probe: init once, before the first parallel region.
    static const bool probe_ready = (par_probe_init(), true);
    (void)probe_ready;

    // Pass A: per-SM work only, so all clusters run flat at full thread width.
    const unsigned n_clusters_flat = m_shader_config->n_simt_clusters;
    const unsigned long long phase_a_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
// guided, per loop: idle SMs make static chunks uneven (a global OMP_SCHEDULE slows others).
#pragma omp parallel for schedule(guided) if(par_a)
    for (unsigned i = 0; i < n_clusters_flat; i++) {
      // Same as get_not_completed() || get_more_cta_left().
      const unsigned not_completed = m_cluster[i]->get_not_completed();
      m_cluster_ran[i] = (not_completed || more_cta_left) ? 1 : 0;   // more_cta_left: hoisted
      if (m_cluster_ran[i]) m_cluster[i]->core_cycle_phase_a();
    }
    if (g_phase_timing_on) {
      g_phase_timers.phase_a_ns += phase_now_ns() - phase_a_wall_start;
      g_phase_timers.phase_a_calls++;
    }

    // Pass B: one chiplet per thread; a chiplet's SMs run in order, which its L2 TLB and GMMU need.
    const unsigned long long phase_b_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
#pragma omp parallel for schedule(runtime) if(par_b)
    for (unsigned ch = 0; ch < n_chiplet_groups; ch++) {
      const unsigned long long phase_b_group_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
      // With an L2 TLB start limit, rotate the first SM each cycle so no SM is always favoured.
      const std::vector<unsigned> &order = m_chiplet_cluster_order[ch];
      const unsigned n_order = (unsigned)order.size();
      const unsigned first =
          (m_shader_config->gpgpu_l2_tlb_starts_per_cycle && n_order)
              ? (unsigned)((gpu_sim_cycle + gpu_tot_sim_cycle) % n_order)
              : 0;
      for (unsigned k = 0; k < n_order; k++) {
        const unsigned ci = (first + k < n_order) ? first + k : first + k - n_order;
        const unsigned i = order[ci];
        if (m_cluster_ran[i]) m_cluster[i]->core_cycle_phase_b();
      }
      if (g_phase_timing_on && ch < 8) {  // [PHASE_TIMING_B]: per-group time + running thread
        const unsigned long long phase_b_group_wall_ns = phase_now_ns() - phase_b_group_wall_start;
        g_phase_timers.phase_b_chip_ns[ch] += phase_b_group_wall_ns;
        g_phase_timers.phase_b_chip_cur_ns[ch] = phase_b_group_wall_ns;  // for the per-cycle maximum
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (tid >= 0 && tid < 16) g_phase_timers.phase_b_chip_thread[ch][tid]++;
      }
    }
    if (g_phase_timing_on) {
      g_phase_timers.phase_b_ns += phase_now_ns() - phase_b_wall_start;
      g_phase_timers.phase_b_calls++;
      {  // [PHASE_TIMING_B2]: per-cycle maximum over groups (serial, after the region joined)
        unsigned long long mx = 0; unsigned am = 0;
        for (unsigned c = 0; c < n_chiplet_groups && c < 8; c++) {
          const unsigned long long v = g_phase_timers.phase_b_chip_cur_ns[c];
          g_phase_timers.phase_b_body_hist[phase_log2us_bucket(v)]++;
          if (v > mx) { mx = v; am = c; }
          g_phase_timers.phase_b_chip_cur_ns[c] = 0;
        }
        g_phase_timers.phase_b_cyclemax_ns += mx;
        g_phase_timers.phase_b_argmax_cnt[am]++;
        g_phase_timers.phase_b_max_hist[phase_log2us_bucket(mx)]++;
      }
    }

    // Pass C is per-SM work only, same as pass A -- flat, full thread width.
    const unsigned long long phase_c_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
// guided, per-loop: same rationale as pass A.
#pragma omp parallel for schedule(guided) reduction(+ : active_sms_this_cycle) if(par_c)
    for (unsigned i = 0; i < n_clusters_flat; i++) {
      if (m_cluster_ran[i]) {
        m_cluster[i]->core_cycle_phase_c();
        // Integer reduction; added to *active_sms once after the loop.
        active_sms_this_cycle += m_cluster[i]->get_n_active_sms();
      }
    }
    if (g_phase_timing_on) {
      g_phase_timers.phase_c_ns += phase_now_ns() - phase_c_wall_start;
      g_phase_timers.phase_c_calls++;
    }
    // AccelWattch core stats. Serial: get_cache_stats() adds into a shared struct.
    if (m_config.g_power_simulation_enabled) {
      for (unsigned i = 0; i < n_clusters_flat; i++) {
        m_cluster[i]->get_icnt_stats(
            m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
        m_cluster[i]->get_cache_stats(
            m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
      }
    }
    // Phase timing "tail": from here to the end of cycle().
    tail_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
#ifdef _OPENMP
    if (pt_guard_active && m_gpu_module &&
        m_gpu_module->get_page_table()->mutation_count() != pt_mut_before) {
      fprintf(stderr,
              "\nFATAL: the GPU-wide page table changed during the parallel "
              "CORE passes\n"
              "       (cycle %llu, %llu changes). Nothing in passes A-C may "
              "change it: first-touch\n"
              "       owners are buffered and committed after pass C. A "
              "page-table write inside\n"
              "       the parallel passes is a simulator bug; rerun with "
              "OMP_NUM_THREADS=1 to find it.\n",
              (unsigned long long)(gpu_sim_cycle + gpu_tot_sim_cycle),
              m_gpu_module->get_page_table()->mutation_count() - pt_mut_before);
      fflush(stderr);
      abort();
    }
#endif
    // Kernels that finished in pass C, in SM order (the 1-thread order).
    commit_deferred_kernel_done();
    // Commit this cycle's deferred first-touch inserts, after the parallel passes and the guard.
    if (m_gpu_module && m_gpu_module->get_page_table())
      m_gpu_module->get_page_table()->commit_deferred_first_touch();
    // Occupancy sampling. Serial: adds into the shared gpu_occupancy struct.
    const unsigned long long occupancy_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->get_current_occupancy(
          gpu_occupancy.aggregate_warp_slot_filled,
          gpu_occupancy.aggregate_theoretical_warp_slots);
    }
    if (g_phase_timing_on) g_phase_timers.tail_occ_ns += phase_now_ns() - occupancy_wall_start;
    *active_sms += (float)active_sms_this_cycle;

    // ACCELSIM_STATEHASH: per-cycle state fingerprint to find where two runs first diverge. Slow.
    static const bool statehash_on = (getenv("ACCELSIM_STATEHASH") != NULL);
    if (statehash_on && m_gpu_module) {
      // FNV over the unit index (a plain sum could cancel out); covers all three page sizes.
      unsigned long long h_l2 = 1469598103934665603ull;
      for (unsigned c = 0; c < m_gpu_module->get_num_chiplets(); c++) {
        const chiplet *ch = m_gpu_module->get_chiplet(c);
        const gpu_l2_tlb *tlbs[3] = {ch->get_l2_tlb_4k(), ch->get_l2_tlb_64k(), ch->get_l2_tlb_2m()};
        for (unsigned k = 0; k < 3; k++) {
          h_l2 ^= (tlbs[k] ? tlbs[k]->state_fingerprint() : 0ull) + (c * 3 + k) * 2654435761ull;
          h_l2 *= 1099511628211ull;
        }
      }
      unsigned long long h_l1 = 1469598103934665603ull;
      unsigned long long h_sm = 1469598103934665603ull;
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        h_l1 ^= m_cluster[i]->xlat_l1_fingerprint() + i * 2654435761ull;
        h_l1 *= 1099511628211ull;
        h_sm ^= (unsigned long long)m_shader_stats->m_num_sim_winsn[i] + i * 2654435761ull;
        h_sm *= 1099511628211ull;
      }
      printf("[STATEHASH] cyc=%llu insn=%llu l2=%llu l1=%llu sm=%llu\n",
             (unsigned long long)(gpu_sim_cycle + gpu_tot_sim_cycle),
             (unsigned long long)gpu_sim_insn, h_l2, h_l1, h_sm);

      // [SMDETAIL] per SM within ACCELSIM_STATEHASH_LO..HI, to find which SM diverged.
      static const char *lo_s = getenv("ACCELSIM_STATEHASH_LO");
      static const char *hi_s = getenv("ACCELSIM_STATEHASH_HI");
      static const unsigned long long win_lo = lo_s ? strtoull(lo_s, 0, 10) : 0;
      static const unsigned long long win_hi = hi_s ? strtoull(hi_s, 0, 10) : 0;
      const unsigned long long now = gpu_sim_cycle + gpu_tot_sim_cycle;
      if (win_hi && now >= win_lo && now <= win_hi) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
          printf("[SMDETAIL] cyc=%llu sm=%u winsn=%u cyc_cnt=%llu l1=%llu\n",
                 now, i, m_shader_stats->m_num_sim_winsn[i],
                 m_shader_stats->shader_cycles[i],
                 m_cluster[i]->xlat_l1_fingerprint());
      }
    }

    float temp = 0;
    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }
    gpu_sim_cycle++;

    if (g_interactive_debugger_enabled) gpgpu_debug();

      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      if (m_config.g_power_simulation_mode == 0) {
        mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                    m_power_stats, m_config.gpu_stat_sample_freq,
                    gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                    gpu_sim_insn, m_config.g_dvfs_enabled);
      }
    }
#endif

    const unsigned long long cta_issue_wall_start = g_phase_timing_on ? phase_now_ns() : 0;
    issue_block2core();
    if (g_phase_timing_on) g_phase_timers.tail_issue_ns += phase_now_ns() - cta_issue_wall_start;
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() == 0) {
          // Skip clusters with no L1D touch since the last invalidate: the call would do nothing.
          if (m_l1_used.empty() || m_l1_used[i]) {
            m_cluster[i]->cache_invalidate();
            if (!m_l1_used.empty()) m_l1_used[i] = 0;
          }
        } else
          all_threads_complete = 0;
      }
    }

    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        printf("Flushed L2 caches...\n");
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
            printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
          }
        }
      }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t days, hrs, minutes, sec;
      time_t curr_time;
      time(&curr_time);
      unsigned long long elapsed_time =
          MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
      if ((elapsed_time - last_liveness_message_time) >=
              m_config.liveness_message_freq &&
          DTRACE(LIVENESS)) {
        days = elapsed_time / (3600 * 24);
        hrs = elapsed_time / 3600 - 24 * days;
        minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        unsigned long long active = 0, total = 0;
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->get_current_occupancy(active, total);
        }
        DPRINTFG(LIVENESS,
                 "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f%% [%llu / %llu]) "
                 "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                 gpu_tot_sim_insn + gpu_sim_insn,
                 (double)gpu_sim_insn / (double)gpu_sim_cycle,
                 float(active) / float(total) * 100, active, total,
                 (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                 (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                 (unsigned)sec, ctime(&curr_time));
        fflush(stdout);
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(stdout);
          printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
          printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
          shader_print_runtime_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(stdout, false);
      }
    }

    if (!(gpu_sim_cycle % 50000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
    // launch device kernel
    gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
  }
  if (g_phase_timing_on) {
    const unsigned long long cycle_wall_end = phase_now_ns();
    if (clock_mask & CORE) g_phase_timers.tail_ns += cycle_wall_end - tail_wall_start;
    g_phase_timers.cycle_ns += cycle_wall_end - cycle_wall_start;
  }
}

void sst_gpgpu_sim::cycle() {
  SST_cycle();
  return;
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    // counter is size_t: a 32-bit one would loop forever at 4 GiB.
    // Decode each 128 B line once when its sectors share a sub-partition (no mapping bit below 7).
    static const bool line_memo =
        m_memory_config->m_address_mapping.sub_partition_fixed_within(7);
    bool have_line = false;
    size_t memo_line = 0;
    unsigned memo_sub = 0;
    for (size_t counter = 0; counter < count; counter += 32) {
      // 64-bit, or the address would be cut to 32 bits. This VA is not translated,
      // so MCM runs use -gpgpu_perf_sim_memcpy 0.
      const size_t wr_addr = dst_start_addr + counter;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      if (!(line_memo && have_line && (wr_addr >> 7) == memo_line)) {
        addrdec_t raw_addr;
        m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
        memo_sub = raw_addr.sub_partition;
        memo_line = wr_addr >> 7;
        have_line = true;
      }
      const unsigned partition_id =
          memo_sub / m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, memo_sub, mask);
    }
  }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }

void sst_gpgpu_sim::SST_gpgpusim_numcores_equal_check(unsigned sst_numcores) {
  if (m_shader_config->n_simt_clusters != sst_numcores) {
    assert(
        "\nSST core is not equal the GPGPU-sim cores. Open gpgpu-sim.config "
        "file and ensure n_simt_clusters"
        "is the same as SST gpu cores.\n" &&
        0);
  } else {
    printf("\nSST GPU core is equal the GPGPU-sim cores = %d\n", sst_numcores);
  }
}

void sst_gpgpu_sim::SST_cycle() {
  // shader core loading (pop from ICNT into core) follows CORE clock
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    static_cast<sst_simt_core_cluster *>(m_cluster[i])->icnt_cycle_SST();

  // L1 cache + shader core pipeline stages
  m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
      m_cluster[i]->core_cycle();
      *active_sms += m_cluster[i]->get_n_active_sms();
    }
    // Update core icnt/cache stats for GPUWattch
    m_cluster[i]->get_icnt_stats(
        m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
        m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
    m_cluster[i]->get_cache_stats(
        m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
  }
  float temp = 0;
  for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
    temp += m_shader_stats->m_pipeline_duty_cycle[i];
  }
  temp = temp / m_shader_config->num_shader();
  *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
  // cout<<"Average pipeline duty cycle: "<<*average_pipeline_duty_cycle<<endl;

  if (g_single_step && ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
    asm("int $03");
  }
  gpu_sim_cycle++;
  if (g_interactive_debugger_enabled) gpgpu_debug();

    // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle,
                gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn,
                m_config.g_dvfs_enabled);
  }
#endif

  issue_block2core();

  if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
    time_t days, hrs, minutes, sec;
    time_t curr_time;
    time(&curr_time);
    unsigned long long elapsed_time =
        MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
    if ((elapsed_time - last_liveness_message_time) >=
        m_config.liveness_message_freq) {
      days = elapsed_time / (3600 * 24);
      hrs = elapsed_time / 3600 - 24 * days;
      minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
      sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

      last_liveness_message_time = elapsed_time;
    }
    visualizer_printstat();
    m_memory_stats->memlatstat_lat_pw();
    if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0)) {
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
          m_memory_partition_unit[i]->print_stat(stdout);
        printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
        printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
      }
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
        shader_print_runtime_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
        shader_print_l1_miss_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
        shader_print_scheduler_stat(stdout, false);
    }
  }

  if (!(gpu_sim_cycle % 20000)) {
    // deadlock detection
    if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
      gpu_deadlock = true;
    } else {
      last_gpu_sim_insn = gpu_sim_insn;
    }
  }
  try_snap_shot(gpu_sim_cycle);
  spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
  // launch device kernel
  gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
}

// Hook for gpu_module to observe mallocs and track virtual address ranges
void gpgpu_sim::on_gpu_malloc(uint64_t va, size_t aligned_size,
                              unsigned page_shift) {
  if (m_gpu_module)
    m_gpu_module->on_malloc(va, aligned_size, page_shift);
}

// Page shift for one allocation from -gpgpu_page_size_policy. 0-2: one size for all
// (small allocations waste a page); 3-5: the largest enabled page the allocation fills.
unsigned gpgpu_sim::decide_page_shift(size_t size) const {
  const unsigned policy = m_shader_config->gpgpu_page_size_policy;
  switch (policy) {
    // ---- fixed ----
    case 0: return GPU_PAGE_SHIFT;   // all 4KB (baseline)
    case 1: return GPU_64K_SHIFT;    // all 64KB
    case 2: return GPU_HUGE_SHIFT;   // all 2MB
    // ---- automatic: largest enabled page with size >= page size ----
    case 3: return (size >= GPU_HUGE_PAGE_SIZE) ? GPU_HUGE_SHIFT   // 4KB/64KB/2MB
                 : (size >= GPU_64K_PAGE_SIZE)  ? GPU_64K_SHIFT
                                                : GPU_PAGE_SHIFT;
    case 4: return (size >= GPU_HUGE_PAGE_SIZE) ? GPU_HUGE_SHIFT   // 4KB / 2MB
                                                : GPU_PAGE_SHIFT;
    case 5: return (size >= GPU_64K_PAGE_SIZE)  ? GPU_64K_SHIFT    // 4KB / 64KB
                                                : GPU_PAGE_SHIFT;
    default:
      fprintf(stderr, "GPGPU-Sim: unknown -gpgpu_page_size_policy %u\n", policy);
      abort();
  }
}
