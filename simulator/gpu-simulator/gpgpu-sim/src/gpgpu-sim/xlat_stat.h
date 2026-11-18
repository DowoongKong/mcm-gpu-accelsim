#pragma once
#include <cstdint>
#include <cstdio>

// XLATLAT: per-(warp, vpn) translation latency breakdown; segments must tile [t_issue, t_done].

// Where a request got its translation (disjoint buckets; only XLAT_B_PTW drove a page walk).
enum xlat_bucket_t {
  XLAT_B_L1_HIT = 0,  // the L1 array already held the VPN when the probe resolved
  XLAT_B_L1_MERGE,    // merged onto an in-flight L1 MSHR entry
  XLAT_B_L2_HIT,      // L1 miss -> the L2 array held it
  XLAT_B_L2_MERGE,    // L1 miss -> merged onto a walk another cluster started
  XLAT_B_PTW,         // L1 miss -> L2 miss -> this request drove the page walk
  XLAT_B_NUM
};

const char *xlat_bucket_name(unsigned b);

// Log2 histogram: bin 0 is exactly zero, bin k >= 1 covers [2^(k-1), 2^k).
enum { XLAT_HIST_BINS = 34 };
static inline unsigned xlat_hist_bin(unsigned long long v) {
  if (v == 0) return 0;
  const unsigned b = 64u - (unsigned)__builtin_clzll(v);
  return b < XLAT_HIST_BINS ? b : XLAT_HIST_BINS - 1;
}

// Print only the non-empty bins.
static inline void xlat_hist_print(FILE *f, const unsigned long long *h) {
  for (unsigned i = 0; i < XLAT_HIST_BINS; i++)
    if (h[i]) fprintf(f, " b%u=%llu", i, h[i]);
}

// How the L2 resolved a request. Decides which XLAT_B_* bucket it lands in.
enum xlat_l2_class_t { XLAT_L2C_HIT = 0, XLAT_L2C_MERGE, XLAT_L2C_MISS };

// What one page walk did, as scalars; the GMMU keeps one per in-flight walk and hands it to the L2.
struct xlat_walk_t {
  uint64_t t_pwq_enq    = 0;  // t5: the request entered the PWQ
  uint64_t t_walk_start = 0;  // t6: a walker picked it up
  uint8_t  n_steps   = 0;     // page-table reads sent to memory (<= 4)
  uint8_t  n_remote  = 0;     // of those, the ones that crossed a chiplet
  uint8_t  n_pwc_hit = 0;     // levels served by the page walk cache
  uint16_t icnt_bp   = 0;     // cycles lost to icnt input backpressure retries
  // L2MISSLOG: walk memory-step timing and fault count, read only by gmmu::complete_walk()'s log.
  uint8_t  n_fault       = 0;  // PT nodes / leaf this walk created (reference mode only)
  uint32_t step_lat_sum  = 0;  // issue -> reply cycles, summed over the memory steps
  uint32_t step_lat_max  = 0;  // the slowest single memory step
  uint64_t t_step_issue  = 0;  // issue cycle of the step currently in memory (0 = none)
  uint64_t last_entry_pa = 0;  // PA of the last page-table entry read from memory
};

// L2 + GMMU boundaries of one (cluster, vpn) request; attached to the L1-miss request that drove it.
struct xlat_down_t {
  uint64_t t_l2_arm  = 0;  // t3: first gate arm for this (cluster, vpn)
  uint64_t t_l2_miss = 0;  // t4: first resolve() verdict (HIT/MERGE collapse here)
  uint64_t t_pwq_enq = 0;  // t5: gmmu->translate() accepted the request
  uint32_t l2_port_polls = 0;  // COUNT of PORT_BUSY answers, not cycles
  uint32_t l2_resv_fail  = 0;  // cycles blocked on an L2 MSHR-full
  uint32_t l2_reprobe    = 0;  // full-latency re-arms after a verdict (stays 0)
  uint8_t  l2_class      = XLAT_L2C_MISS;

  // GMMU side (scalars; per-level timing is kept as a GMMU-wide distribution).
  uint64_t t_walk_start = 0;  // t6: a walker picked the request out of the PWQ
  uint64_t t_walk_done  = 0;  // t7: the walk produced a PTE
  uint8_t  n_steps   = 0;     // page-table reads that went to memory (<= 4)
  uint8_t  n_remote  = 0;     // of those, the ones that crossed a chiplet
  uint8_t  n_pwc_hit = 0;     // levels served by the page walk cache
  uint16_t icnt_bp   = 0;     // cycles the walk lost to icnt input backpressure
};

// Timestamps are absolute core cycles (gpu_sim_cycle + gpu_tot_sim_cycle); 0 = not reached yet.
struct xlat_req_rec_t {
  uint64_t t_issue = 0;            // t0: first access() call for this (warp, vpn)
  uint64_t t_l1_inject_first = 0;  // t1 of the FIRST epoch (before any re-issue)
  uint64_t t_l1_inject = 0;        // t1 of the CURRENT epoch
  uint64_t t_l1_resolve = 0;       // t2: probe reached the pipe head and resolved
  uint64_t t_fill = 0;             // t8: the VPN landed in this SM's L1 array
  uint64_t t_done = 0;             // t9: access() handed the PTE to the warp

  uint32_t l1_resv_fail = 0;   // sub-attribution inside S2: MSHR-full re-injects
  uint32_t l1_port_polls = 0;  // COUNT of PORT_BUSY answers, NOT cycles (see below)
  uint8_t  bucket = XLAT_B_L1_HIT;
  uint8_t  has_down = 0;
  xlat_down_t down;
};

// Accumulated breakdown for one bucket; completed records are folded in and dropped.
struct xlat_accum_t {
  unsigned long long n = 0;
  unsigned long long total = 0;  // t_done - t_issue
  unsigned long long s0 = 0;     // re-issue epochs after an L1 eviction
  unsigned long long s1 = 0;     // issue -> first injection into the lookup pipe
  unsigned long long s2 = 0;     // last injection -> probe resolved
  unsigned long long smid = 0;   // probe resolved -> array filled (UNSPLIT total)
  unsigned long long s3 = 0;     // L1 miss handed off -> L2 armed its probe
  unsigned long long s4 = 0;     // L2 probe (+ MSHR-full retries) -> verdict
  unsigned long long s5 = 0;     // L2 miss -> PWQ accepted (PWQ-full re-probes)
  unsigned long long rest = 0;   // PWQ wait + page walk + fill delivery (UNSPLIT)
  unsigned long long s6 = 0;     // PWQ queueing delay
  unsigned long long s7 = 0;     // page walk service time
  unsigned long long s8 = 0;     // walk done -> this SM's L1 array filled
  unsigned long long n_walk = 0;   // requests whose rest was split into s6/s7/s8
  unsigned long long steps = 0;    // page-table memory reads, summed
  unsigned long long remote = 0;   // of those, off-chiplet
  unsigned long long pwc_hit = 0;  // levels served by the PWC
  unsigned long long icnt_bp = 0;
  unsigned long long s9 = 0;     // array filled -> warp replayed and took the PTE
  unsigned long long l1_resv_fail = 0;
  unsigned long long l1_port_polls = 0;
  unsigned long long l2_resv_fail = 0;
  unsigned long long l2_port_polls = 0;
  unsigned long long l2_reprobe = 0;
  unsigned long long unsplit = 0;   // smid kept whole (merged request, or no down)
  unsigned long long residual = 0;  // MUST stay 0; any other value invalidates the run

  // Histograms of s6 and s7, only for requests that drove a walk (they match n_walk).
  unsigned long long hist_s6[XLAT_HIST_BINS] = {};  // PWQ queueing delay
  unsigned long long hist_s7[XLAT_HIST_BINS] = {};  // walk service time
};

// Fold one completed record into acc; returns its residual. smid is split only for the request that
// owns the downstream path. Port polls count answers, not cycles (one parked warp replays per cycle).
static inline unsigned long long xlat_fold(xlat_accum_t &acc,
                                           const xlat_req_rec_t &r) {
  const unsigned long long total = r.t_done - r.t_issue;
  const unsigned long long s0 = r.t_l1_inject - r.t_l1_inject_first;
  const unsigned long long s1 = r.t_l1_inject_first - r.t_issue;
  const unsigned long long s2 = r.t_l1_resolve - r.t_l1_inject;
  const unsigned long long smid = r.t_fill - r.t_l1_resolve;
  const unsigned long long s9 = r.t_done - r.t_fill;

  const bool splittable =
      r.has_down && r.down.t_l2_arm >= r.t_l1_resolve &&
      r.down.t_l2_miss >= r.down.t_l2_arm &&
      r.down.t_pwq_enq >= r.down.t_l2_miss && r.t_fill >= r.down.t_pwq_enq;

  if (splittable) {
    acc.s3 += r.down.t_l2_arm  - r.t_l1_resolve;
    acc.s4 += r.down.t_l2_miss - r.down.t_l2_arm;
    acc.s5 += r.down.t_pwq_enq - r.down.t_l2_miss;
    acc.rest += r.t_fill - r.down.t_pwq_enq;
    acc.l2_resv_fail  += r.down.l2_resv_fail;
    acc.l2_port_polls += r.down.l2_port_polls;
    acc.l2_reprobe    += r.down.l2_reprobe;

    // Only a request that drove a walk splits `rest`: read s6/s7/s8 against n_walk, not n.
    const bool walked = r.down.t_walk_start >= r.down.t_pwq_enq &&
                        r.down.t_walk_done  >= r.down.t_walk_start &&
                        r.t_fill            >= r.down.t_walk_done &&
                        r.down.t_walk_start != 0;
    if (walked) {
      const unsigned long long q = r.down.t_walk_start - r.down.t_pwq_enq;
      const unsigned long long sv = r.down.t_walk_done - r.down.t_walk_start;
      acc.s6 += q;
      acc.s7 += sv;
      acc.s8 += r.t_fill - r.down.t_walk_done;
      acc.hist_s6[xlat_hist_bin(q)]++;
      acc.hist_s7[xlat_hist_bin(sv)]++;
      acc.n_walk++;
      acc.steps   += r.down.n_steps;
      acc.remote  += r.down.n_remote;
      acc.pwc_hit += r.down.n_pwc_hit;
      acc.icnt_bp += r.down.icnt_bp;
    }
  } else {
    acc.unsplit++;
  }

  const unsigned long long sum = s0 + s1 + s2 + smid + s9;
  const unsigned long long residual = (total >= sum) ? (total - sum) : (sum - total);

  acc.n++;
  acc.total += total;
  acc.s0 += s0;
  acc.s1 += s1;
  acc.s2 += s2;
  acc.smid += smid;
  acc.s9 += s9;
  acc.l1_resv_fail += r.l1_resv_fail;
  acc.l1_port_polls += r.l1_port_polls;
  acc.residual += residual;
  return residual;
}

// True when every boundary was observed; requests still in flight at the end are counted apart.
static inline bool xlat_rec_complete(const xlat_req_rec_t &r) {
  return r.t_issue && r.t_l1_inject_first && r.t_l1_inject && r.t_l1_resolve &&
         r.t_fill && r.t_done;
}
// ---------------------------------------------------------------------------
// XLATSUM: readable summary rows from the raw accumulators ([_XLATLAT] is the fixed raw format).
// avg == svc + wait. *_mshr_full spans all epochs; its *_mshr_full_reissue part lies in reissue.
struct xlat_sum_t {
  unsigned long long n = 0;
  // raw sums (request-weighted), folded straight from xlat_accum_t
  unsigned long long total=0, s0=0, s1=0, s2=0, smid=0, s3=0, s4=0, s5=0,
                     rest=0, s6=0, s7=0, s8=0, s9=0, l1rf=0, l2rf=0;
  // GPU-wide s6/s7 histograms; n_walk is the bin total, as in xlat_accum_t.
  unsigned long long n_walk = 0;
  unsigned long long hist_s6[XLAT_HIST_BINS] = {};
  unsigned long long hist_s7[XLAT_HIST_BINS] = {};
  void add(const xlat_accum_t &a) {
    n += a.n; total += a.total; s0 += a.s0; s1 += a.s1; s2 += a.s2;
    smid += a.smid; s3 += a.s3; s4 += a.s4; s5 += a.s5; rest += a.rest;
    s6 += a.s6; s7 += a.s7; s8 += a.s8; s9 += a.s9;
    l1rf += a.l1_resv_fail; l2rf += a.l2_resv_fail;
    n_walk += a.n_walk;
    for (unsigned i = 0; i < XLAT_HIST_BINS; i++) {
      hist_s6[i] += a.hist_s6[i]; hist_s7[i] += a.hist_s7[i];
    }
  }
  void add(const xlat_sum_t &o) {
    n += o.n; total += o.total; s0 += o.s0; s1 += o.s1; s2 += o.s2;
    smid += o.smid; s3 += o.s3; s4 += o.s4; s5 += o.s5; rest += o.rest;
    s6 += o.s6; s7 += o.s7; s8 += o.s8; s9 += o.s9;
    l1rf += o.l1rf; l2rf += o.l2rf;
    n_walk += o.n_walk;
    for (unsigned i = 0; i < XLAT_HIST_BINS; i++) {
      hist_s6[i] += o.hist_s6[i]; hist_s7[i] += o.hist_s7[i];
    }
  }
  // Print the s6 (PWQ wait) and s7 (walk service) histograms; nothing if no request walked.
  void print_hist(FILE *f, const char *kind, const char *head, const char *out) const {
    if (!n_walk) return;
    fprintf(f, "[XLATHIST] %s kind=%s out=%s hist=s6_pwqwait n=%llu", head, kind, out, n_walk);
    xlat_hist_print(f, hist_s6);
    fprintf(f, "\n[XLATHIST] %s kind=%s out=%s hist=s7_walk n=%llu", head, kind, out, n_walk);
    xlat_hist_print(f, hist_s7);
    fprintf(f, "\n");
  }
  // One row. head = e.g. "SM=12" or "GPU", out = bucket name or "ALL".
  void print(FILE *f, const char *kind, const char *head, const char *out) const {
    if (!n) return;
    const double N = (double)n;
    // l1rf spans all re-issue epochs but s2 only the last, so clamp it to s2.
    const unsigned long long l1rf_in_s2 = l1rf < s2 ? l1rf : s2;
    const unsigned long long l2rf_in_s4 = l2rf < s4 ? l2rf : s4;
    const double l1_probe = (double)(s2 - l1rf_in_s2) / N;
    const double l2_probe = (double)(s4 - l2rf_in_s4) / N;
    const double walk_svc = (double)s7 / N;
    const double merge_wait = (double)(smid - (s3 + s4 + s5 + rest)) / N;
    const double pwq_wait = (double)s6 / N, fill_return = (double)s8 / N;
    // The part of rest not split into s6+s7+s8 (merged L2 requests) is wait, not service.
    const double rest_unwalked = (double)(rest - (s6 + s7 + s8)) / N;
    const double svc = l1_probe + l2_probe + walk_svc;
    // The l1rf excess lies in s0, already in wait: report it apart instead of adding it again.
    const double l1rf_reissue = (double)(l1rf - l1rf_in_s2) / N;
    const double l2rf_reissue = (double)(l2rf - l2rf_in_s4) / N;
    // svc + wait == total / N exactly: every raw term appears once.
    const double wait =
        (double)(s0 + s1 + l1rf_in_s2 + s3 + l2rf_in_s4 + s5 + s6 + s8 + s9) / N +
        merge_wait + rest_unwalked;
    fprintf(f,
      "[XLATSUM] %s kind=%s out=%s n=%llu avg=%.1f svc=%.1f wait=%.1f | "
      "l1_enter_wait=%.1f l1_probe=%.1f l1_mshr_full=%.1f merge_wait=%.1f "
      "link_l1l2=%.1f l2_probe=%.1f l2_mshr_full=%.1f pwq_admit=%.1f "
      "pwq_wait=%.1f walk_svc=%.1f fill_return=%.1f rest_merged=%.1f "
      "reissue=%.1f replay_wait=%.1f l1_mshr_full_reissue=%.1f "
      "l2_mshr_full_reissue=%.1f\n",
      head, kind, out, n, (double)total / N, svc, wait,
      (double)s1 / N, l1_probe, (double)l1rf / N, merge_wait,
      (double)s3 / N, l2_probe, (double)l2rf / N, (double)s5 / N,
      pwq_wait, walk_svc, fill_return, rest_unwalked,
      (double)s0 / N, (double)s9 / N, l1rf_reissue, l2rf_reissue);
  }
};
