// par_serial_probe.h -- DIAGNOSTIC ONLY, not part of the performance model.
//
// Purpose: the parallel CORE phase reproduces the sequential trajectory at 1/2/4
// threads but diverges at 8 and 16 threads, and ThreadSanitizer has so far only
// pointed at statistics counters.  To locate the subsystem that carries the
// missing happens-before edge, this header lets ONE subsystem be serialized at
// run time, without rebuilding, via
//
//     ACCELSIM_SERIAL_SUBSYS=icnt|fetch|issue|mf     (comma separated)
//
// Reading of the result:
//   probe ON  and 16 threads now reproduce the 1-thread digest -> the race is
//                 inside the region the probe covers.
//   probe ON  and 16 threads still diverge                     -> it is not
//                 (solely) inside that region.
//
// An omp_lock_t is used rather than "#pragma omp critical" because the choice
// must be made at run time; the lock establishes the same release/acquire edges
// a critical section would, so a race fully contained in the region really is
// removed, not merely made less likely.
//
// When no subsystem is selected every guard compiles down to one predictable
// branch on a cached bool, so the default build is unaffected.

#ifndef PAR_SERIAL_PROBE_H
#define PAR_SERIAL_PROBE_H

// The last three together partition phase C exactly --
//   cycle_phase_c() = execute_c() + read_operands() + issue() + N x (decode() +
//   fetch()) -- so if each of them comes back negative while
//   ACCELSIM_SERIAL_PHASE=C comes back positive, the conflicting accesses sit in
//   two DIFFERENT sub-regions, which is itself a result.  PAR_PROBE_ICNT cuts
//   across all of them and is listed first because 3 of the 4 top same-phase
//   TSan sites are on the icnt push path.
enum par_probe_subsys {
  PAR_PROBE_ICNT = 0,   // icnt_push / icnt_has_buffer / icnt_pop entry points
  PAR_PROBE_FETCH,      // shader_core_ctx::decode() + fetch(): L1I cycle
  PAR_PROBE_ISSUE,      // shader_core_ctx::issue(): issue_warp inst copy
  PAR_PROBE_EXEC,       // shader_core_ctx::execute_c() + read_operands()
  PAR_PROBE_N
};

// Must be called once, from the serial region, before any parallel region is
// entered.  Reads the environment variable and initialises the locks.
void par_probe_init();

bool par_probe_enabled(par_probe_subsys s);
void par_probe_lock(par_probe_subsys s);
void par_probe_unlock(par_probe_subsys s);

// Scoped guard.  Holds no lock and costs one branch when the subsystem is off.
class par_probe_guard {
 public:
  explicit par_probe_guard(par_probe_subsys s)
      : m_subsys(s), m_held(par_probe_enabled(s)) {
    if (m_held) par_probe_lock(m_subsys);
  }
  ~par_probe_guard() {
    if (m_held) par_probe_unlock(m_subsys);
  }

 private:
  par_probe_guard(const par_probe_guard &);
  par_probe_guard &operator=(const par_probe_guard &);
  par_probe_subsys m_subsys;
  bool m_held;
};

#endif  // PAR_SERIAL_PROBE_H
