// Copyright (c) 2019, Mahmoud Khairy
// Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#ifndef _LOCAL_INTERCONNECT_HPP_
#define _LOCAL_INTERCONNECT_HPP_

#include <iostream>
#include <map>
#include <queue>
#include <vector>
using namespace std;

enum Interconnect_type { REQ_NET = 0, REPLY_NET = 1 };

enum Arbiteration_type { NAIVE_RR = 0, iSLIP = 1 };

struct inct_config {
  // config for local interconnect
  unsigned in_buffer_limit;
  unsigned out_buffer_limit;
  unsigned subnets;
  Arbiteration_type arbiter_algo;
  unsigned verbose;
  unsigned grant_cycles;
  unsigned use_ring;   // 1 = inter-chiplet ring, 0 = direct handoff
  unsigned ring_buf;   // ring per-link / eject buffer depth (packets)
  unsigned ring_hop_latency;  // ring per-hop latency in cycles (paper: 32)
  int      ring_hop_latency_ptw;  // same, for page-table-walk packets; <0 = inherit
  unsigned ring_link_width;  // flits drained per cycle per link (BW = LW * flit * freq)
  unsigned xbar_port_width;  // intra-chiplet xbar: flits drained per cycle per OUTPUT port
                             // (per-port BW = xbar_port_width * flit_size * icnt_freq).
};

// P-5: (temp) diagnostic-counter gate (defined in gpu-sim.cc); the inline
// ring code below uses it to skip RINGLAT_/RINGLINK_/BUCKET_ accounting.
extern const bool g_accelsim_diag;

class xbar_router {
 public:
  xbar_router(unsigned router_id, enum Interconnect_type m_type,
              unsigned n_shader, unsigned n_mem,
              const struct inct_config& m_localinct_config,
              unsigned n_icp = 0,
              bool inter_via_ring = false,  // true -> xbar uses LOCAL flit size only
              bool both_dirs = false);      // H1-A: one router carries REQ+REPLY
                                            // (-icnt_subnets 1) -> every node is
                                            // a destination candidate
  ~xbar_router();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  void* Pop(unsigned ouput_deviceID);
  void Advance();
  // H3-3b (speed): 출력 포트 out 에 패킷을 넣을 때마다 켤 미러 바이트(없으면 nullptr).
  // Advance() 는 칩렛 x 서브넷 병렬이지만 한 출력 포트는 한 라우터에만 있으므로 바이트마다
  // 쓰는 스레드는 하나다.
  void set_out_mirror(unsigned out, unsigned char *slot) { m_out_mirror[out] = slot; }
  bool out_empty(unsigned out) const { return out_buffers[out].empty(); }

  bool Busy() const;
  bool Has_Buffer_In(unsigned input_deviceID, unsigned size,
                     bool update_counter = false);
  bool Has_Buffer_Out(unsigned output_deviceID, unsigned size);

  // Inter-chiplet port (ICP) helpers, used by LocalInterconnect::handoff().
  // Inspect the ICP egress out-buffer front without popping it.
  void* TopOut(unsigned output_deviceID) const;
  bool  HasOut(unsigned output_deviceID) const;
  // Flit count of one mem_fetch, using the same rule as Push(); for buffer checks.
  unsigned flits_of(void* data) const;
  // ICP is the last node index in this router (clusters, then mems, then ICP).
  unsigned icp_index() const { return _n_shader + _n_mem; }
  // H1-A: is `o` one of the outputs a packet on this router can be addressed to?
  // Used by the XBAR_CAND_CHECK assert in Push(); cheap enough to keep public.
  bool is_out_candidate(unsigned o) const {
    for (unsigned c : m_out_cand) if (c == o) return true;
    return false;
  }
  bool     has_icp() const { return _n_icp != 0; }

  // Fix B: ICP ingress VOQ (per-destination-output queues) — removes head-of-line
  // blocking of the single ICP ingress FIFO. Shared budget = in_buffer_limit (D1).
  bool has_icp_voq_room(unsigned flits) const {
    return m_icp_voq_flits + flits <= in_buffer_limit;
  }
  void push_icp_voq(unsigned out, void* data, unsigned flits) {
    m_icp_voq[out].push(Packet(data, out, flits));
    m_icp_voq_flits += flits;
  }

  // some stats
  unsigned long long cycles;
  unsigned long long conflicts;
  unsigned long long conflicts_util;
  unsigned long long cycles_util;
  unsigned long long reqs_util;
  unsigned long long out_buffer_full;
  unsigned long long out_buffer_util;
  unsigned long long in_buffer_full;
  unsigned long long in_buffer_util;
  unsigned long long packets_num;

 private:
  void iSLIP_Advance();
  void RR_Advance();

  struct Packet {
    Packet(void* m_data, unsigned m_output_deviceID, unsigned m_flits) {
      data = m_data;
      output_deviceID = m_output_deviceID;
      remaining_flits = m_flits;
      orig_flits = m_flits;  // BUG-9 fix: kept so the egress push can restore
                             // the packet's full size for out-buffer occupancy
    }

    void* data;
    unsigned output_deviceID;
    unsigned remaining_flits;
    unsigned orig_flits;
  };
  // 44-1: FIFO with an O(1) running total of remaining_flits. queue_flits()
  // used to COPY the whole std::queue and walk it on every Has_Buffer_* call
  // (up to nodes x port_width times per xbar per icnt cycle). Invariant:
  // flits == sum of remaining_flits over q, kept at the three mutation points
  // (push / pop / dec_front_flits). Plain data, no virtuals, no allocation
  // beyond the queue itself.
  struct flit_queue {
    queue<Packet> q;
    unsigned flits;
    flit_queue() : flits(0) {}
    bool empty() const { return q.empty(); }
    size_t size() const { return q.size(); }
    const Packet& front() const { return q.front(); }
    void push(const Packet& p) { q.push(p); flits += p.remaining_flits; }
    void pop() { flits -= q.front().remaining_flits; q.pop(); }
    // The only in-place mutation of a queued packet (iSLIP serves one flit).
    void dec_front_flits() { q.front().remaining_flits--; flits--; }
#ifdef XBAR_FLITS_CHECK
    unsigned recount() const {
      queue<Packet> c = q; unsigned s = 0;
      while (!c.empty()) { s += c.front().remaining_flits; c.pop(); }
      return s;
    }
#endif
  };
  vector<flit_queue> in_buffers;
  vector<flit_queue> out_buffers;
  vector<unsigned char*> m_out_mirror;   // H3-3b: [local out] -> mirror byte or nullptr
  void mark_out(unsigned out) { if (m_out_mirror[out]) *m_out_mirror[out] = 1; }
  // Fix B: per-output-destination ICP ingress queues (VOQ) + shared-budget flit count
  // + per-output RR turn (alternate VOQ vs local inputs, starvation-free).
  vector<queue<Packet> > m_icp_voq;
  unsigned m_icp_voq_flits;
  vector<unsigned> m_icp_voq_rr;
  unsigned _n_shader, _n_mem, total_nodes;
  unsigned in_buffer_limit, out_buffer_limit;
  vector<unsigned> next_node;  // used for iSLIP arbit
  unsigned next_node_id;       // used for RR arbit
  // H1-A (speed): outputs that can EVER be a packet's destination on this router.
  // Local index layout is clusters [0,_n_shader), mems [_n_shader,_n_shader+_n_mem),
  // then the ICP. Push() sends a local packet to m_local_idx[dst] and a remote one to
  // icp_index(), and handoff_ring() ejects to m_local_idx[final_dev] -- so in REQ_NET
  // a destination is a mem or the ICP, and in REPLY_NET a cluster or the ICP. The other
  // outputs can never be granted (nothing ever carries their index) and their
  // out_buffers stay empty, so Has_Buffer_Out() never fails on them either: skipping
  // them in the grant loop changes no state and no counter. Ascending order, so the
  // surviving outputs keep the order the full 0..total_nodes loop visited them in.
  vector<unsigned> m_out_cand;
  // H1-B (speed): per-output count of inputs whose FRONT packet is addressed to that
  // output. Arbitration only ever inspects an input's front, so a zero here (with an
  // empty ICP VOQ) means the O(total_nodes) input scan cannot grant -- the scan is
  // then skipped, which is what removes the empty-scan cost. Rebuilt once per
  // iSLIP_Advance() inside the conflict-count pass that already walks every input,
  // and kept exact for the rest of that cycle by the delta update at the pop site.
  // Deliberately NOT maintained from Push(): Push() is reachable from several SM
  // threads at once, and a shared counter there would be a data race (the per-cycle
  // rebuild inside Advance() needs no synchronisation at all).
  vector<unsigned> m_out_pending;
  // H1-D (speed): per-cycle "already seen this destination" marks for the conflict
  // count, replacing a heap-allocated vector + std::find scan. Stamp equality means
  // "seen this cycle", so no clearing pass is needed.
  vector<unsigned long long> m_seen_stamp;
  unsigned long long m_seen_cur;
  unsigned m_id;
  enum Interconnect_type router_type;
  unsigned active_in_buffers, active_out_buffers;
  Arbiteration_type arbit_type;
  unsigned verbose;

  unsigned grant_cycles;
  unsigned grant_cycles_count;

  unsigned m_port_width;  // flits drained per cycle per OUTPUT port (>=1). Sets the
                          // per-port intra-chiplet BW = m_port_width * flit * icnt_freq.
  unsigned _n_icp;  // number of inter-chiplet ports on this router (0 or 1)
  bool m_inter_via_ring;  // when true, do NOT inflate remote flits (ring owns inter BW)

 public:
  unsigned queue_flits(const flit_queue& q) const;  // 44-1: O(1)
  
  friend class LocalInterconnect;
};

// Inter-chiplet ring. One instance per subnet. Bandwidth (link, 1 flit/cycle)
// and wire latency (hop_latency cycles, non-blocking) are modeled separately.
class ring_net {
 public:
  ring_net(unsigned n_nodes, unsigned buf_depth, unsigned hop_latency,
           unsigned hop_latency_ptw, unsigned link_width = 1)
      : N(n_nodes), m_buf(buf_depth), m_hop_latency(hop_latency), 
        m_hop_latency_ptw(hop_latency_ptw),
        m_link_width(link_width ? link_width : 1), m_cycle(0) {
    link.assign(N, std::vector<std::queue<RingPkt> >(2));  // [node][dir] 0=cw,1=ccw (bandwidth)
    wire.assign(N, std::vector<std::queue<RingPkt> >(2));  // [node][dir] (wire latency, in-flight)
    wire_ptw.assign(N, std::vector<std::queue<RingPkt> >(2));  // same, page-table-walk class
    eject_buf.assign(N, std::queue<RingPkt>());
    link_flits.assign(N, std::vector<unsigned long long>(2, 0));   // RINGLINK_ (temp)
    link_occ_sum.assign(N, std::vector<unsigned long long>(2, 0)); // RINGLINK_ (temp)
    for (unsigned k = 0; k < RING_CLS; ++k) {   // 0 = data, 1 = page-table walk
      ringlat_src_sum[k].assign(N, 0);   // RINGSRC_ (temp)
      ringlat_src_cnt[k].assign(N, 0);   // RINGSRC_ (temp)
      bkt_bw_sum[k].assign(N, 0);        // BUCKET_ (temp)
      bkt_ej_sum[k].assign(N, 0);        // BUCKET_ (temp)
    }
  }
  // Inject at node src toward dst. false (no-op) if the outgoing link buffer is full.
  bool inject(unsigned src, unsigned dst, unsigned final_dev, void* data, unsigned flits,
              unsigned cls) {
    unsigned dir = dir_of(src, dst);
    if (link[src][dir].size() >= m_buf) return false;
    RingPkt p; p.data = data; p.final_dev = final_dev; p.dst = dst;
    p.dir = dir; p.flits = flits; p.remaining = flits; p.arrival = 0;
    p.cls = cls; p.wire_seq = 0;   // seq is stamped when the packet enters the wire stage
    p.src = src;                                    /*RINGSRC_*/
    p.inject_cycle = m_cycle;                       /*RINGLAT_*/
    p.enter_link = m_cycle;                         /*BUCKET_*/
    p.enter_wire = 0; p.t_bw_sum = 0; p.t_ej_sum = 0; /*BUCKET_*/
    link[src][dir].push(p);
    return true;
  }
  // One cycle: (A) bandwidth on each link head, (B) wire-latency delivery.
  void advance() {
    ++m_cycle;
    if (g_accelsim_diag) ++link_occ_samples;   // RINGLINK_ (temp, P-5 gated)
    // Phase A: drain up to m_link_width flits per link per cycle, spanning packets
    // (so small packets can pass multiple-per-cycle -> high link bandwidth).
    // BW(per link/dir) = m_link_width * inter_flit_size * freq.
    for (unsigned c = 0; c < N; ++c)
      for (unsigned d = 0; d < 2; ++d) {
        if (g_accelsim_diag)
          link_occ_sum[c][d] += link[c][d].size();  // RINGLINK_ (temp, P-5 gated)
        unsigned budget = m_link_width;
        while (budget > 0 && !link[c][d].empty()) {
          RingPkt& p = link[c][d].front();
          unsigned take = (p.remaining < budget) ? p.remaining : budget;
          if (g_accelsim_diag) link_flits[c][d] += take;  // RINGLINK_ (temp, P-5 gated)
          p.remaining -= take;
          budget      -= take;
          if (p.remaining == 0) {              // finished on this link -> wire (latency) stage
            if (g_accelsim_diag) {                   /*BUCKET_ (P-5 gated)*/
              p.t_bw_sum += (m_cycle - p.enter_link);
              p.enter_wire = m_cycle;
            }
            p.arrival  = m_cycle + hop_lat_of(p.cls);
            p.wire_seq = ++m_wire_seq;
            (p.cls ? wire_ptw[c][d] : wire[c][d]).push(p);
            link[c][d].pop();
          } else {
            break;                              // budget exhausted mid-packet
          }
        }
      }
    // Phase B: deliver wire packets whose arrival<=now to the next node. A delivered
    // packet enters link[next] (xmit) with remaining=flits, so it is never advanced
    // twice in one cycle (Phase A already ran) -> deterministic.
    for (unsigned c = 0; c < N; ++c)
      for (unsigned d = 0; d < 2; ++d) {
        while (std::queue<RingPkt>* wq = wire_head(c, d)) {
          RingPkt p = wq->front();
          unsigned next = neighbor(c, d);
          // BUCKET_ (temp, P-5 gated): wait beyond the legit hop latency here. The
          // legit latency is per class, so a PTW packet is not credited with the data
          // class's wire time (that would make its ejwait come out negative-biased).
          const unsigned hl = hop_lat_of(p.cls);
          unsigned long long ejw = (g_accelsim_diag && m_cycle - p.enter_wire > hl)
                                   ? (m_cycle - p.enter_wire - hl) : 0;
          if (next == p.dst) {                          // arrived at destination chiplet
            if (eject_buf[next].size() >= m_buf) break; // eject full -> wait
            if (g_accelsim_diag) {  /*RINGLAT_/RINGSRC_/BUCKET_ (P-5 gated)*/
              unsigned long long rt = m_cycle - p.inject_cycle;
              ringlat_sum[p.cls] += rt; ringlat_cnt[p.cls]++;
              if (rt > ringlat_max[p.cls]) ringlat_max[p.cls] = rt;
              p.t_ej_sum += ejw;
              if (p.src < ringlat_src_sum[p.cls].size()) {
                ringlat_src_sum[p.cls][p.src] += rt; ringlat_src_cnt[p.cls][p.src]++;
                bkt_bw_sum[p.cls][p.src] += p.t_bw_sum;
                bkt_ej_sum[p.cls][p.src] += p.t_ej_sum;
              }
            }
            xlatdbg_wire_note(p.cls, m_cycle - p.enter_wire);   // XLATDBG_ (temp)
            eject_buf[next].push(p); wq->pop();
          } else {                                      // forward to next hop's link
            if (link[next][d].size() >= m_buf) break;   // downstream link full -> wait
            if (g_accelsim_diag) {                      /*BUCKET_ (P-5 gated)*/
              p.t_ej_sum += ejw;
              p.enter_link = m_cycle;
            }
            xlatdbg_wire_note(p.cls, m_cycle - p.enter_wire);   // XLATDBG_ (temp)
            p.remaining = p.flits;                      // re-transmit (bandwidth) on next link
            link[next][d].push(p); wq->pop();
          }
        }
      }
  }
  bool     has_eject(unsigned c) const { return !eject_buf[c].empty(); }
  void*    top_eject(unsigned c) const { return eject_buf[c].front().data; }
  unsigned eject_final_dev(unsigned c) const { return eject_buf[c].front().final_dev; }
  void     pop_eject(unsigned c)       { eject_buf[c].pop(); }
  bool busy() const {
    for (unsigned c = 0; c < N; ++c) {
      if (!eject_buf[c].empty()) return true;
      for (unsigned d = 0; d < 2; ++d)
        if (!link[c][d].empty() || !wire[c][d].empty() || !wire_ptw[c][d].empty())
          return true;
    }
    return false;
  }
 public:
  // RINGLAT_ (temp): ring traversal-time stats (inject -> eject), inter-chiplet only.
  // Indexed by packet class: [0] = data, [1] = page-table walk. Once the two
  // classes can carry different hop latencies (-icnt_ptw_ring_hop_latency), one
  // combined average is a blend of two populations and cannot answer "did data
  // traffic get faster?". The combined value is still reported, recomputed as the
  // sum over classes, so existing logs stay comparable.
  enum { RING_CLS = 2 };
  // XLATDBG_ (temp): per-hop wire residency, MEASURED directly as
  // (delivery cycle - wire entry cycle), not derived by subtracting other stats.
  // A packet that is never blocked leaves the wire stage exactly hop_lat_of(cls)
  // cycles after entering it, so `min` must equal the configured latency for that
  // class. A larger min means the wire stage still blocks one class behind the
  // other; avg > min is ordinary eject/link backpressure and is reported by ejwait.
  // Recorded only under ACCELSIM_DIAG, because enter_wire is stamped only there.
  unsigned long long xlatdbg_wire_sum[RING_CLS] = {0, 0};
  unsigned long long xlatdbg_wire_cnt[RING_CLS] = {0, 0};
  unsigned long long xlatdbg_wire_min[RING_CLS] = {~0ULL, ~0ULL};
  unsigned long long xlatdbg_wire_max[RING_CLS] = {0, 0};
  void xlatdbg_wire_note(unsigned cls, unsigned long long w) {
    if (!g_accelsim_diag) return;
    xlatdbg_wire_sum[cls] += w; xlatdbg_wire_cnt[cls]++;
    if (w < xlatdbg_wire_min[cls]) xlatdbg_wire_min[cls] = w;
    if (w > xlatdbg_wire_max[cls]) xlatdbg_wire_max[cls] = w;
  }
  // Configured latency for a class, so the report can print the expected value
  // beside the measured one instead of the reader having to look it up.
  unsigned hop_lat_cfg(unsigned cls) const { return hop_lat_of(cls); }
  unsigned long long ringlat_sum[RING_CLS] = {0, 0};
  unsigned long long ringlat_cnt[RING_CLS] = {0, 0};
  unsigned long long ringlat_max[RING_CLS] = {0, 0};
  // RINGLINK_ (temp): per-(node,dir) link load. link_flits = flits transmitted (throughput),
  // link_occ_sum/link_occ_samples = avg queue occupancy. Shows tie-break CW/CCW imbalance.
  std::vector<std::vector<unsigned long long> > link_flits;
  std::vector<std::vector<unsigned long long> > link_occ_sum;
  unsigned long long link_occ_samples = 0;
  unsigned ring_N() const { return N; }
  // RINGSRC_ (temp): per-SOURCE-chiplet ring residency (inject->eject). Isolates
  // whether a source's ring time is high (e.g. eject-order unfairness at the hub).
  std::vector<unsigned long long> ringlat_src_sum[RING_CLS];
  std::vector<unsigned long long> ringlat_src_cnt[RING_CLS];
  // BUCKET_ (temp): per-source ring-residency breakdown. t_wire = total - bw - ejwait.
  //   bw     = time in link[] bandwidth stage (serialization/queue for the link)
  //   ejwait = time in wire[] AFTER arrival passed but blocked from eject/forward
  //            (this is the artifact shadow: ICP-ingress/downstream backpressure).
  std::vector<unsigned long long> bkt_bw_sum[RING_CLS];
  std::vector<unsigned long long> bkt_ej_sum[RING_CLS];
 private:
  struct RingPkt { void* data; unsigned final_dev, dst, dir, flits, remaining, src;
                   unsigned cls;                 // packet class: 0 = data, 1 = page-table walk
                   unsigned long long wire_seq;  // stamped on wire entry; orders the class merge
                   unsigned long long arrival;
                   unsigned long long inject_cycle; /*RINGLAT_*/
                   unsigned long long enter_link, enter_wire; /*BUCKET_*/
                   unsigned long long t_bw_sum, t_ej_sum;     /*BUCKET_*/ };
  unsigned N, m_buf, m_hop_latency;
  unsigned m_hop_latency_ptw;          // wire latency for the page-table-walk class
  unsigned long long m_wire_seq = 0;   // monotonic, stamped when a packet enters the wire
  unsigned m_link_width;  // flits/cycle per link
  unsigned long long m_cycle;
  mutable unsigned m_tie_rr = 0;  // round-robin cursor for equidistant direction split
  std::vector<std::vector<std::queue<RingPkt> > > link;  // bandwidth stage [node][dir]
  std::vector<std::vector<std::queue<RingPkt> > > wire;  // latency stage   [node][dir]
  std::vector<std::vector<std::queue<RingPkt> > > wire_ptw;  // latency stage, PTW class
  std::vector<std::queue<RingPkt> >               eject_buf;
  unsigned hop_lat_of(unsigned cls) const {
    return cls ? m_hop_latency_ptw : m_hop_latency;
  }
  // Earliest deliverable packet on (c,d): (arrival, wire_seq) order. The sequence
  // number is stamped on wire entry, so when both classes share one hop latency the
  // merged order is exactly the order the single wire FIFO used to deliver in.
  std::queue<RingPkt>* wire_head(unsigned c, unsigned d) {
    std::queue<RingPkt>* best = NULL;
    std::queue<RingPkt>* qs[2] = { &wire[c][d], &wire_ptw[c][d] };
    for (unsigned i = 0; i < 2; ++i) {
      std::queue<RingPkt>* q = qs[i];
      if (q->empty() || q->front().arrival > m_cycle) continue;
      if (!best || q->front().arrival < best->front().arrival ||
          (q->front().arrival == best->front().arrival &&
           q->front().wire_seq < best->front().wire_seq))
        best = q;
    }
    return best;
  }
  unsigned neighbor(unsigned c, unsigned d) const {
    return (d == 0) ? (c + 1) % N : (c + N - 1) % N;     // 0=cw, 1=ccw
  }
  unsigned dir_of(unsigned s, unsigned d) const {        // shortest path direction
    unsigned cw = (d + N - s) % N, ccw = (s + N - d) % N;
    if (cw < ccw) return 0u;                              // strictly shorter CW
    if (ccw < cw) return 1u;                              // strictly shorter CCW
    // Equidistant (distance N/2, e.g. 0<->2 on a 4-node ring): a fixed CW tie-break
    // overloads one direction (2-hop traffic always rides the CW link out of the hub).
    // Deterministic round-robin -> split equidistant traffic 50/50 across CW/CCW.
    // (A queue-occupancy adaptive rule is near-useless here: with a wide link_width the
    //  bandwidth-stage queue drains almost every cycle, so its size ~0 and never tips.)
    return (m_tie_rr++ & 1u);
  }
};

class LocalInterconnect {
 public:
  LocalInterconnect(const struct inct_config& m_localinct_config);
  ~LocalInterconnect();
  static LocalInterconnect* New(const struct inct_config& m_inct_config);
  void CreateInterconnect(unsigned n_shader, unsigned n_mem);

  // node side functions
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  void* Pop(unsigned ouput_deviceID);
  void Advance();
  bool Busy() const;
  bool HasBuffer(unsigned deviceID, unsigned int size) const;
  // H3-3b: Pop(dev) 이 읽는 바로 그 출력 버퍼에 대한 미러 등록 / 비었는지 확인.
  void SetOutMirror(unsigned dev, unsigned char *slot);
  bool OutEmpty(unsigned dev) const;
  void DisplayStats() const;
  void DisplayOverallStats() const;
  unsigned GetFlitSize() const;

  void DisplayState(FILE* fp) const;

  void configure_chiplets(unsigned n_chiplets,
                          const std::vector<unsigned>& cluster_to_chiplet,   // size n_shader
                          const std::vector<unsigned>& mem_sub_to_chiplet);  // size n_mem

 protected:
  const inct_config& m_inct_config;

  unsigned n_shader, n_mem;
  unsigned n_subnets;

  // One local crossbar per chiplet: net[subnet][chiplet]. (Phase 1)
  std::vector<std::vector<xbar_router*> > net;

  // Inter-chiplet egress -> ingress transfer. Zero latency in Phase 1;
  // the ring is inserted here in Phase 2.
  void handoff();

  // Device->chiplet owner maps and O(1) lookups, built by configure_chiplets().
  unsigned m_n_chiplets = 1;
  bool     m_chiplets_configured = false;
  std::vector<unsigned> m_dev_chiplet;      // [n_shader+n_mem] -> chiplet id
  std::vector<unsigned> m_local_idx;        // [n_shader+n_mem] -> local node idx in its chiplet
  std::vector<unsigned> m_n_local_cluster;  // [n_chiplets] cluster count per chiplet
  std::vector<unsigned> m_n_local_mem;      // [n_chiplets] mem-sub count per chiplet

  bool m_use_ring = false;                 // ring vs Phase-1 direct handoff
  std::vector<ring_net*> m_ring;           // one ring per subnet (when m_use_ring)
  void handoff_ring();                     // ICP egress<->ring<->ICP ingress

  // RINGINJ_ (temp): per-(subnet,src chiplet) source-side ring egress accounting.
  // egress_busy: cycles the icp egress had >=1 packet pending at handoff time.
  // inj_stall:   cycles inject() failed (ring link buffer full) -> ring-BW bound.
  // inj_ok:      successful injects. handoff_advances: total handoff_ring() calls.
  std::vector<std::vector<unsigned long long> > m_rinj_egress_busy;
  std::vector<std::vector<unsigned long long> > m_rinj_stall;
  std::vector<std::vector<unsigned long long> > m_rinj_ok;
  unsigned long long m_rinj_advances = 0;

  // Lookups (within each chiplet router: clusters first, then mems, ICP last).
  unsigned dev_chiplet(unsigned dev) const { return m_dev_chiplet[dev]; }
  unsigned local_idx(unsigned dev)   const { return m_local_idx[dev]; }
  unsigned icp_idx(unsigned ch)      const { return m_n_local_cluster[ch] + m_n_local_mem[ch]; }
};

#endif
