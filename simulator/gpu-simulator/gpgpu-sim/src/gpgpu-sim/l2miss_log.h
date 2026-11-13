// L2MISSLOG (s29, 2026-11-11): runtime-gated timeline of L2 TLB misses and
// page walks for the ref-vs-base timing debug. OFF unless the environment
// variable ACCELSIM_L2MISS_LOG names an output file; then one line per event:
//   M <cycle> <chiplet> <cluster> 0x<vpn> <kind> <ft>
//       L2 TLB verdict that needs a walk. kind 0 = MISS (a new walk starts),
//       kind 1 = MERGE (MSHR hit, joins the walk in flight). ft = 1 when the
//       page has no present leaf PTE at this moment (first touch; reference
//       mode only, a pinned run always prints 0).
//   C <cycle> <chiplet> 0x<vpn> <t_pwq_enq> <t_walk_start> <steps> <remote>
//     <pwc_hits> <faults> <icnt_bp> <step_lat_sum> <step_lat_max>
//     0x<last_entry_pa> 0x<data_pa>
//       a walk completed (L2 TLB fill). steps = page-table reads that went to
//       memory, remote = those that crossed a chiplet, faults = nodes/leaf this
//       walk created, step_lat_* = issue-to-reply cycles of the memory steps,
//       last_entry_pa = PA of the last page-table entry read, data_pa = the
//       translated PA (frame of the data page).
// Cycles are absolute core cycles (gpu_sim_cycle + gpu_tot_sim_cycle), the
// same clock as the XLATLAT timestamps. The log only reads state that already
// exists and writes nothing back, so a run with the log on is cycle-identical
// to a run with it off (gate L2M-1 in docs/s29_l2miss_timeline_refbase_debug.ko.md).
#ifndef L2MISS_LOG_H
#define L2MISS_LOG_H
#include <cstdio>
// Returns the open log file, or NULL when logging is off. Defined in gmmu.cc.
FILE* l2miss_log_file();
#endif
