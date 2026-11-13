// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include <csignal>
#include <execinfo.h>
#include <unistd.h>

// XLATDBG: print a backtrace on SIGSEGV. gdb cannot catch this particular crash --
// it changes the process environment (and therefore the heap/stack layout) and the
// fault does not reproduce under it. Handling the signal in-process keeps the
// layout byte-identical to the failing run.
static void xlatdbg_segv_handler(int sig) {
  void* frames[40];
  int n = backtrace(frames, 40);
  const char msg[] = "\n*** XLATDBG SIGSEGV backtrace ***\n";
  ssize_t w = write(STDERR_FILENO, msg, sizeof(msg) - 1); (void)w;
  backtrace_symbols_fd(frames, n, STDERR_FILENO);
  signal(sig, SIG_DFL);
  raise(sig);
}

#include "accel-sim.h"

/* TO DO:
 * NOTE: the current version of trace-driven is functionally working fine,
 * but we still need to improve traces compression and simulation speed.
 * This includes:
 *
 * 1- Prefetch concurrent thread that prefetches traces from disk (to not be
 * limited by disk speed)
 *
 * 2- traces compression format a) cfg format and remove
 * thread/block Id from the head and b) using zlib library to save in binary
 * format
 *
 * 3- Efficient memory improvement (save string not objects - parse only 10 in
 * the buffer)
 *
 * 4- Seeking capability - thread scheduler (save tb index and warp
 * index info in the traces header)
 *
 * 5- Get rid off traces intermediate files -
 * change the tracer
 */

int main(int argc, const char **argv) {
  signal(SIGSEGV, xlatdbg_segv_handler);  // XLATDBG

  accel_sim_framework accel_sim(argc, argv);
  accel_sim.simulation_loop();

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}
