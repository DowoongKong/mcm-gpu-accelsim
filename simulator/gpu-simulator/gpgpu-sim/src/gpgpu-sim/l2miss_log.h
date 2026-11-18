// ACCELSIM_L2MISS_LOG=<file>: timeline of L2 TLB misses (M lines) and completed page walks (C lines).
// Read-only (a run with it on is cycle-identical); line format in simulator/README.md (Debug logs).
#ifndef L2MISS_LOG_H
#define L2MISS_LOG_H
#include <cstdio>
// Returns the open log file, or NULL when logging is off. Defined in gmmu.cc.
FILE* l2miss_log_file();
#endif
