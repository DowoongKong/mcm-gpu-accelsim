#pragma once
#include <cstdint>
#include <cstdio>

// XLATLAT: per-request address-translation latency breakdown.
//
// One record per (warp, vpn) translation request, living in gpu_l1_tlb while the
// request is in flight and folded into accumulators when it completes. The
// design doc is docs/xlat_latency_breakdown_plan.ko.md; the segment names below
// match its section 3-1.
//
// The segments must TILE the interval [t_issue, t_done] exactly: a non-zero
// residual means a transition was missed and every breakdown derived from it is
// unreliable. That invariant is checked on every close-out.
//
// Stage 2 resolves the L1 boundaries and the L2 boundaries; what remains lumped
// is `rest` (PWQ wait + page walk + fill delivery), which stage 3 splits.

// Where a request ultimately got its translation. Buckets are disjoint and are
// reported separately: only XLAT_B_PTW requests actually drove a page walk, so
// they are the ones whose downstream segments belong to them (see design doc
// 3-2, primary vs merged).
enum xlat_bucket_t {
  XLAT_B_L1_HIT = 0,  // the L1 array already held the VPN when the probe resolved
  XLAT_B_L1_MERGE,    // merged onto an in-flight L1 MSHR entry
  XLAT_B_L2_HIT,      // L1 miss -> the L2 array held it
  XLAT_B_L2_MERGE,    // L1 miss -> merged onto a walk another cluster started
  XLAT_B_PTW,         // L1 miss -> L2 miss -> this request drove the page walk
  XLAT_B_NUM
};

const char *xlat_bucket_name(unsigned b);

// Log2 histogram. The queueing model assumes M/G/32: `G` says the walk service
// time follows some general distribution and `M` says arrivals are Poisson.
// Neither has ever been checked -- a mean cannot check either -- so the shape is
// collected here. Bin 0 is exactly zero; bin k >= 1 covers [2^(k-1), 2^k).
// One __builtin_clzll per sample, so this is effectively free.
enum { XLAT_HIST_BINS = 34 };
static inline unsigned xlat_hist_bin(unsigned long long v) {
  if (v == 0) return 0;
  const unsigned b = 64u - (unsigned)__builtin_clzll(v);
  return b < XLAT_HIST_BINS ? b : XLAT_HIST_BINS - 1;
}

// Print a histogram compactly: only non-empty bins, so a mostly-empty
// distribution costs a few characters instead of 34 zeros per line.
static inline void xlat_hist_print(FILE *f, const unsigned long long *h) {
  for (unsigned i = 0; i < XLAT_HIST_BINS; i++)
    if (h[i]) fprintf(f, " b%u=%llu", i, h[i]);
}

// How the L2 resolved a request. Decides which XLAT_B_* bucket it lands in.
enum xlat_l2_class_t { XLAT_L2C_HIT = 0, XLAT_L2C_MERGE, XLAT_L2C_MISS };

// What one page walk did, in scalars. The GMMU keeps one of these per in-flight
// walk (folded into the start-cycle map it already had, so no extra map), and
// hands it to the L2 on completion. Per-level detail is intentionally absent:
// the queueing model needs the walk duration and the remote/local split
// -- E[S] = E[M]*t_local + E[R]*t_icnt -- and holding levels here would mean
// copying an array into every request waiting on the walk.
struct xlat_walk_t {
  uint64_t t_pwq_enq    = 0;  // t5: the request entered the PWQ
  uint64_t t_walk_start = 0;  // t6: a walker picked it up
  uint8_t  n_steps   = 0;     // page-table reads sent to memory (<= 4)
  uint8_t  n_remote  = 0;     // of those, the ones that crossed a chiplet
  uint8_t  n_pwc_hit = 0;     // levels served by the page walk cache
  uint16_t icnt_bp   = 0;     // cycles lost to icnt input backpressure retries
  // L2MISSLOG (s29): per-walk memory-step timing and the fault flag. Written by
  // the GMMU, read only by the gated log line in gmmu::complete_walk(); no
  // statistic or decision consumes them.
  uint8_t  n_fault       = 0;  // PT nodes / leaf this walk created (reference mode only)
  uint32_t step_lat_sum  = 0;  // issue -> reply cycles, summed over the memory steps
  uint32_t step_lat_max  = 0;  // the slowest single memory step
  uint64_t t_step_issue  = 0;  // issue cycle of the step currently in memory (0 = none)
  uint64_t last_entry_pa = 0;  // PA of the last page-table entry read from memory
};

// Downstream (L2 + GMMU) boundaries for ONE (cluster, vpn) request. Produced by
// gpu_l2_tlb, carried back to the requesting SM through the existing fill path,
// and attached only to the L1-miss request that actually drove it.
struct xlat_down_t {
  uint64_t t_l2_arm  = 0;  // t3: first gate arm for this (cluster, vpn)
  uint64_t t_l2_miss = 0;  // t4: first resolve() verdict (HIT/MERGE collapse here)
  uint64_t t_pwq_enq = 0;  // t5: gmmu->translate() accepted the request
  uint32_t l2_port_polls = 0;  // COUNT of PORT_BUSY answers, not cycles
  uint32_t l2_resv_fail  = 0;  // cycles blocked on an L2 MSHR-full
  uint32_t l2_reprobe    = 0;  // full-latency re-arms caused by a PWQ-full rollback
  uint8_t  l2_class      = XLAT_L2C_MISS;

  // GMMU side. Scalars only: the queueing model needs the walk's DURATION and
  // how many of its page-table reads went off-chiplet -- E[S] = E[M]*t_local +
  // E[R]*t_icnt -- not the latency of each individual level. Per-level timing is
  // kept as a GMMU-wide distribution instead, so nothing is copied per request.
  uint64_t t_walk_start = 0;  // t6: a walker picked the request out of the PWQ
  uint64_t t_walk_done  = 0;  // t7: the walk produced a PTE
  uint8_t  n_steps   = 0;     // page-table reads that went to memory (<= 4)
  uint8_t  n_remote  = 0;     // of those, the ones that crossed a chiplet
  uint8_t  n_pwc_hit = 0;     // levels served by the page walk cache
  uint16_t icnt_bp   = 0;     // cycles the walk lost to icnt input backpressure
};

// Timestamps are absolute core cycles (gpu_sim_cycle + gpu_tot_sim_cycle), the
// same clock shader.cc and gmmu.cc use. 0 means "not reached yet"; cycle 0 never
// carries a real request because translation cannot complete on the first cycle.
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

// Accumulated breakdown for one bucket. Completed records are folded in here and
// destroyed; nothing is retained per request.
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
  unsigned long long s6 = 0;     // PWQ queueing delay  <- the hypothesis' target
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

  // Shape of the two quantities the queueing model rests on. Filled only for
  // requests that actually drove a walk, so they line up with n_walk.
  unsigned long long hist_s6[XLAT_HIST_BINS] = {};  // PWQ queueing delay
  unsigned long long hist_s7[XLAT_HIST_BINS] = {};  // walk service time -- the `G`
};

// Fold one completed record into acc. Returns the residual for this record so the
// caller can assert on it in verification builds.
//
// S1 uses the FIRST injection and S2 the LAST, with S0 covering the gap between
// them, so the five top-level segments stay disjoint even when a request is
// re-issued after its fill was evicted from the small L1 array before the warp
// consumed it.
//
// smid is split into s3/s4/s5/rest only when this request owns the downstream
// path AND the timestamps are monotone. A merged request waits on someone else's
// work, so splitting its smid with the owner's timestamps would produce negative
// segments; those stay lumped and are counted in `unsplit`.
//
// l1_port_polls / l2_port_polls are deliberately NOT segments. A parked warp does
// not call access() every cycle -- the ldst unit replays at most one parked warp
// per cycle -- so the number of PORT_BUSY answers cannot be read as a cycle count.
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

    // `rest` splits further only for a request that actually drove a walk. An
    // L2 array hit never reaches the GMMU, so its walk timestamps stay 0 and it
    // is simply not counted in n_walk -- s6/s7/s8 must be read against n_walk,
    // never against n.
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

// Is the record complete enough to fold? Every boundary must have been observed;
// a request still in flight when the simulation ends is counted separately rather
// than folded with zeroed timestamps.
static inline bool xlat_rec_complete(const xlat_req_rec_t &r) {
  return r.t_issue && r.t_l1_inject_first && r.t_l1_inject && r.t_l1_resolve &&
         r.t_fill && r.t_done;
}
// ---------------------------------------------------------------------------
// 34: XLATSUM -- self-describing summary rows built on top of the raw
// accumulators. The raw [_XLATLAT] lines are the digest-stable machine format
// and never change; these rows are additive ([XLATSUM] matches no digest
// pattern) and exist so a human gets the true weighted average and a GPU-wide
// view without hand-folding 256 SMs x 5 outcomes.
//
// Naming (raw -> summary):
//   s1              l1_enter_wait   entry into the L1 lookup pipe (ports/replay)
//   s2 - l1_resv    l1_probe        fixed L1 array probe latency
//   l1_resv_fail    l1_mshr_full    head retries against a full L1 MSHR
//   s3              link_l1l2       L1->L2 queue + L2 entry (incl. L2 port gate)
//   s4 - l2_resv    l2_probe        fixed L2 array probe latency
//   l2_resv_fail    l2_mshr_full
//   s5              pwq_admit       PWQ-full re-probes
//   s6/s7/s8 (/n)   pwq_wait / walk_svc / fill_return
//   smid unsplit    merge_wait      waiting on someone else's walk
//   s0 / s9         reissue / replay_wait
//   l1rf - min(..)  l1_mshr_full_reissue   MSHR-full retry cycles that fall in an
//                   earlier re-issue epoch, i.e. inside `reissue`, not inside the
//                   last epoch's probe. Reported so nothing is dropped; it is NOT
//                   added to wait again (that was the BUG-2 double count).
// Identity: avg == svc + wait (guaranteed by the residual==0 tiling).
// NOTE: the printed components sum to wait only after subtracting
// l1_mshr_full_reissue from l1_mshr_full (and likewise for l2), because
// l1_mshr_full keeps its original meaning (all epochs) for log compatibility.
struct xlat_sum_t {
  unsigned long long n = 0;
  // raw sums (request-weighted), folded straight from xlat_accum_t
  unsigned long long total=0, s0=0, s1=0, s2=0, smid=0, s3=0, s4=0, s5=0,
                     rest=0, s6=0, s7=0, s8=0, s9=0, l1rf=0, l2rf=0;
  // ST-7: GPU-wide shape of the two queueing-model quantities, folded from the
  // per-SM hist_s6/hist_s7 that used to be printed per SM. n_walk is the bin
  // total, exactly as in xlat_accum_t.
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
  // ST-7: the S6 (PWQ wait) and S7 (walk service) distributions for this
  // aggregate. Skipped when nothing walked. Bins as in xlat_hist_bin().
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
    // Re-issued records (s0 > 0) accumulate l1rf across epochs while s2 spans
    // only the last epoch, so the difference can go negative; clamp for the
    // display and fold the excess into the wait side (it is stall time).
    const unsigned long long l1rf_in_s2 = l1rf < s2 ? l1rf : s2;
    const unsigned long long l2rf_in_s4 = l2rf < s4 ? l2rf : s4;
    const double l1_probe = (double)(s2 - l1rf_in_s2) / N;
    const double l2_probe = (double)(s4 - l2rf_in_s4) / N;
    const double walk_svc = (double)s7 / N;
    const double merge_wait = (double)(smid - (s3 + s4 + s5 + rest)) / N;
    const double pwq_wait = (double)s6 / N, fill_return = (double)s8 / N;
    // rest tiles into s6+s7+s8 only for walk owners; the un-walked remainder of rest
    // (merged L2 requests waiting on the owner's walk) is wait, not service.
    const double rest_unwalked = (double)(rest - (s6 + s7 + s8)) / N;
    const double svc = l1_probe + l2_probe + walk_svc;
    // l1rf accumulates MSHR-full retry cycles over EVERY re-issue epoch, while s2
    // spans only the last epoch, so l1rf can exceed s2. The excess happened during
    // the earlier epochs, whose wall time is s0 -- and wait already counts s0 in
    // full. Adding the unclamped l1rf here would therefore count the excess twice
    // and break the identity below (observed: PTW bucket off by up to 1.9%). Use
    // the same clamped value the probe split used, and report the excess under its
    // own key rather than dropping it.
    const double l1rf_reissue = (double)(l1rf - l1rf_in_s2) / N;
    const double l2rf_reissue = (double)(l2rf - l2rf_in_s4) / N;
    // Algebra check: svc + wait == total/N exactly. s2 splits into
    // (s2 - l1rf_in_s2) + l1rf_in_s2 and s4 likewise; merge_wait/rest_unwalked are
    // the unsplit remainders of smid and rest; every raw term appears exactly once.
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
