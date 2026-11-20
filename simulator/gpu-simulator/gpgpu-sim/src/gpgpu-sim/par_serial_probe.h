// par_serial_probe.h -- diagnostic only: ACCELSIM_SERIAL_SUBSYS=icnt|fetch|issue|mf serializes one
// subsystem of the parallel CORE phase at run time (omp lock), to locate a 1-thread vs N-thread divergence.

#ifndef PAR_SERIAL_PROBE_H
#define PAR_SERIAL_PROBE_H

// FETCH, ISSUE and EXEC together cover phase C; ICNT cuts across them.
enum par_probe_subsys {
  PAR_PROBE_ICNT = 0,   // env token "icnt":  icnt_push / has_buffer / pop entry points
  PAR_PROBE_FETCH,      // env token "fetch": shader_core_ctx::decode() + fetch(): L1I cycle
  PAR_PROBE_ISSUE,      // env token "issue": shader_core_ctx::issue(): issue_warp inst copy
  PAR_PROBE_EXEC,       // env token "mf": execute_c() + read_operands()
  PAR_PROBE_N           // token strings live in g_probe_names (par_serial_probe.cc)
};

// Call once from the serial region, before any parallel region: reads the env, sets up the locks.
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
