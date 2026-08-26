// par_serial_probe.cc -- see par_serial_probe.h for what this is for.

#include "par_serial_probe.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

const char *const g_probe_names[PAR_PROBE_N] = {"icnt", "fetch", "issue", "mf"};

omp_lock_t g_probe_locks[PAR_PROBE_N];
bool g_probe_on[PAR_PROBE_N] = {false, false, false, false};
bool g_probe_ready = false;

// Matches a comma separated list exactly, so that "mf" does not match a future
// name that merely contains it.
bool selected(const char *list, const char *name) {
  const size_t n = strlen(name);
  for (const char *p = list; *p != '\0';) {
    const char *end = strchr(p, ',');
    const size_t len = (end != NULL) ? (size_t)(end - p) : strlen(p);
    if (len == n && strncmp(p, name, n) == 0) return true;
    if (end == NULL) break;
    p = end + 1;
  }
  return false;
}

}  // namespace

void par_probe_init() {
  if (g_probe_ready) return;
  const char *env = getenv("ACCELSIM_SERIAL_SUBSYS");
  for (int i = 0; i < PAR_PROBE_N; i++) {
    omp_init_lock(&g_probe_locks[i]);
    g_probe_on[i] = (env != NULL) && selected(env, g_probe_names[i]);
    if (g_probe_on[i])
      printf("[PARPROBE] serializing subsystem '%s'\n", g_probe_names[i]);
  }
  if (env == NULL || env[0] == '\0')
    printf("[PARPROBE] no subsystem serialized (control)\n");
  fflush(stdout);
  g_probe_ready = true;
}

bool par_probe_enabled(par_probe_subsys s) { return g_probe_on[s]; }

void par_probe_lock(par_probe_subsys s) { omp_set_lock(&g_probe_locks[s]); }

void par_probe_unlock(par_probe_subsys s) { omp_unset_lock(&g_probe_locks[s]); }
