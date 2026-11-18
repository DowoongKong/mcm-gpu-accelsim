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
  unsigned ring_hop_latency;  // ring per-hop wire latency, ICNT cycles
  int      ring_hop_latency_ptw;  // same, for page-table-walk packets; <0 = inherit
  unsigned ring_link_width;  // flits drained per cycle per link (BW = LW * flit * freq)
  unsigned xbar_port_width;  // intra-chiplet xbar: flits drained per cycle per OUTPUT port
};

// True under ACCELSIM_DIAG (gpu-sim.cc); only then does the ring update its diagnostic counters.
extern const bool g_accelsim_diag;

class xbar_router {
 public:
  xbar_router(unsigned router_id, enum Interconnect_type m_type,
              unsigned n_shader, unsigned n_mem,
              const struct inct_config& m_localinct_config,
              unsigned n_icp = 0,
              bool inter_via_ring = false,  // true -> xbar uses LOCAL flit size only
              bool both_dirs = false);      // one router carries REQ+REPLY (-icnt_subnets 1)
  ~xbar_router();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  void* Pop(unsigned ouput_deviceID);
  void Advance();
  // Busy flag set when a packet enters output `out` (nullptr = none); one router thread writes it.
  void set_out_mirror(unsigned out, unsigned char *slot) { m_out_mirror[out] = slot; }
  bool out_empty(unsigned out) const { return out_buffers[out].empty(); }

  bool Busy() const;
  bool Has_Buffer_In(unsigned input_deviceID, unsigned flits,
                     bool update_counter = false);
  bool Has_Buffer_Out(unsigned output_deviceID, unsigned size);

  // Inter-chiplet port (ICP) helpers for handoff(). TopOut() peeks the egress front without popping.
  void* TopOut(unsigned output_deviceID) const;
  bool  HasOut(unsigned output_deviceID) const;
  // Flit count of one mem_fetch, using the same rule as Push(); for buffer checks.
  unsigned flits_of(void* data) const;
  // ICP is the last node index in this router (clusters, then mems, then ICP).
  unsigned icp_index() const { return _n_shader + _n_mem; }
  // Can output `o` be a packet's destination on the router? Used by the XBAR_CAND_CHECK assert.
  bool is_out_candidate(unsigned o) const {
    for (unsigned c : m_out_cand) if (c == o) return true;
    return false;
  }
  bool     has_icp() const { return _n_icp != 0; }

  // ICP ingress VOQ: one queue per output (no head-of-line blocking), sharing one in_buffer_limit.
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
      orig_flits = m_flits;  // full size, restored for out-buffer occupancy
    }

    void* data;
    unsigned output_deviceID;
    unsigned remaining_flits;
    unsigned orig_flits;
  };
  // FIFO that keeps flits == sum of remaining_flits, so queue_flits() is O(1).
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
  vector<unsigned char*> m_out_mirror;   // [local out] -> busy flag or nullptr
  void mark_out(unsigned out) { if (m_out_mirror[out]) *m_out_mirror[out] = 1; }
  // ICP ingress VOQ, its flit total, and a per-output turn alternating VOQ and local inputs.
  vector<queue<Packet> > m_icp_voq;
  unsigned m_icp_voq_flits;
  vector<unsigned> m_icp_voq_rr;
  unsigned _n_shader, _n_mem, total_nodes;
  unsigned in_buffer_limit, out_buffer_limit;
  vector<unsigned> next_node;  // used for iSLIP arbit
  unsigned next_node_id;       // used for RR arbit
  // Outputs that can ever be a destination here, ascending: mems + ICP in REQ_NET, clusters + ICP
  // in REPLY_NET. Skipping the others in the grant loop is exact: nothing is addressed to them.
  vector<unsigned> m_out_cand;
  // Per output: inputs whose front packet goes there; 0 with an empty VOQ skips the input scan.
  // Rebuilt in iSLIP_Advance(), not kept in Push(): Push() runs on several SM threads at once.
  vector<unsigned> m_out_pending;
  // Per-cycle seen-destination marks for the conflict count (stamp == m_seen_cur means seen).
  vector<unsigned long long> m_seen_stamp;
  unsigned long long m_seen_cur;
  unsigned m_id;
  enum Interconnect_type router_type;
  unsigned active_in_buffers, active_out_buffers;
  Arbiteration_type arbit_type;
  unsigned verbose;

  unsigned grant_cycles;
  unsigned grant_cycles_count;

  unsigned m_port_width;  // flits drained per cycle per OUTPUT port (>=1)
  unsigned _n_icp;  // number of inter-chiplet ports on this router (0 or 1)
  bool m_inter_via_ring;  // when true, do NOT inflate remote flits (ring owns inter BW)

 public:
  unsigned queue_flits(const flit_queue& q) const;  // O(1)
  
  friend class LocalInterconnect;
};

// Inter-chiplet ring, one per subnet: a link sends link_width flits per cycle, then the wire adds
// hop_latency. The wire stage is unbounded, so a blocked delivery never stalls the link.
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
    link_flits.assign(N, std::vector<unsigned long long>(2, 0));   // [RINGLINK]
    link_occ_sum.assign(N, std::vector<unsigned long long>(2, 0)); // [RINGLINK]
    for (unsigned k = 0; k < RING_CLS; ++k) {   // 0 = data, 1 = page-table walk
      ringlat_src_sum[k].assign(N, 0);   // [RINGSRC]
      ringlat_src_cnt[k].assign(N, 0);   // [RINGSRC]
      bkt_bw_sum[k].assign(N, 0);        // [RINGSRC] bw/ejwait
      bkt_ej_sum[k].assign(N, 0);        // [RINGSRC] bw/ejwait
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
    p.src = src;                                    /*[RINGSRC]*/
    p.inject_cycle = m_cycle;                       /*[RINGLAT]*/
    p.enter_link = m_cycle;                         /*[RINGSRC] bw/ejwait*/
    p.enter_wire = 0; p.t_bw_sum = 0; p.t_ej_sum = 0; /*[RINGSRC] bw/ejwait*/
    link[src][dir].push(p);
    return true;
  }
  // One cycle: (A) bandwidth on each link head, (B) wire-latency delivery.
  void advance() {
    ++m_cycle;
    if (g_accelsim_diag) ++link_occ_samples;   // [RINGLINK]
    // Phase A: each link drains up to m_link_width flits per cycle, across packet boundaries.
    for (unsigned c = 0; c < N; ++c)
      for (unsigned d = 0; d < 2; ++d) {
        if (g_accelsim_diag)
          link_occ_sum[c][d] += link[c][d].size();  // [RINGLINK]
        unsigned budget = m_link_width;
        while (budget > 0 && !link[c][d].empty()) {
          RingPkt& p = link[c][d].front();
          unsigned take = (p.remaining < budget) ? p.remaining : budget;
          if (g_accelsim_diag) link_flits[c][d] += take;  // [RINGLINK]
          p.remaining -= take;
          budget      -= take;
          if (p.remaining == 0) {              // finished on this link -> wire (latency) stage
            if (g_accelsim_diag) {                   /*[RINGSRC] bw/ejwait*/
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
    // Phase B: deliver arrived wire packets; runs after Phase A, so no packet moves twice a cycle.
    for (unsigned c = 0; c < N; ++c)
      for (unsigned d = 0; d < 2; ++d) {
        while (std::queue<RingPkt>* wq = wire_head(c, d)) {
          RingPkt p = wq->front();
          unsigned next = neighbor(c, d);
          // [RINGSRC] ejwait: wait beyond the packet class's own hop latency.
          const unsigned hl = hop_lat_of(p.cls);
          unsigned long long ejw = (g_accelsim_diag && m_cycle - p.enter_wire > hl)
                                   ? (m_cycle - p.enter_wire - hl) : 0;
          if (next == p.dst) {                          // arrived at destination chiplet
            if (eject_buf[next].size() >= m_buf) break; // eject full -> wait
            if (g_accelsim_diag) {  /*[RINGLAT], [RINGSRC]*/
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
            xlatdbg_wire_note(p.cls, m_cycle - p.enter_wire);   // [XLATDBG_WIRE]
            eject_buf[next].push(p); wq->pop();
          } else {                                      // forward to next hop's link
            if (link[next][d].size() >= m_buf) break;   // downstream link full -> wait
            if (g_accelsim_diag) {                      /*[RINGSRC] bw/ejwait*/
              p.t_ej_sum += ejw;
              p.enter_link = m_cycle;
            }
            xlatdbg_wire_note(p.cls, m_cycle - p.enter_wire);   // [XLATDBG_WIRE]
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
  // [RINGLAT]: ring inject -> eject time per packet class ([0] data, [1] page-table walk).
  enum { RING_CLS = 2 };
  // [XLATDBG_WIRE]: measured per-hop wire time per class (ACCELSIM_DIAG only). An unblocked
  // packet stays exactly hop_lat_of(cls), so min must equal the configured latency.
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
  // Configured hop latency of a class, printed beside the measured one.
  unsigned hop_lat_cfg(unsigned cls) const { return hop_lat_of(cls); }
  unsigned long long ringlat_sum[RING_CLS] = {0, 0};
  unsigned long long ringlat_cnt[RING_CLS] = {0, 0};
  unsigned long long ringlat_max[RING_CLS] = {0, 0};
  // [RINGLINK]: per-(node,dir) flits sent and queue occupancy sum (avg = sum / samples).
  std::vector<std::vector<unsigned long long> > link_flits;
  std::vector<std::vector<unsigned long long> > link_occ_sum;
  unsigned long long link_occ_samples = 0;
  unsigned ring_N() const { return N; }
  // [RINGSRC]: ring time (inject -> eject) per source chiplet.
  std::vector<unsigned long long> ringlat_src_sum[RING_CLS];
  std::vector<unsigned long long> ringlat_src_cnt[RING_CLS];
  // [RINGSRC] per source: bw = time in link[], ejwait = time blocked in wire[] (wire = the rest).
  std::vector<unsigned long long> bkt_bw_sum[RING_CLS];
  std::vector<unsigned long long> bkt_ej_sum[RING_CLS];
 private:
  struct RingPkt { void* data; unsigned final_dev, dst, dir, flits, remaining, src;
                   unsigned cls;                 // packet class: 0 = data, 1 = page-table walk
                   unsigned long long wire_seq;  // stamped on wire entry; orders the class merge
                   unsigned long long arrival;
                   unsigned long long inject_cycle; /*[RINGLAT]*/
                   unsigned long long enter_link, enter_wire; /*[RINGSRC] bw/ejwait*/
                   unsigned long long t_bw_sum, t_ej_sum;     /*[RINGSRC] bw/ejwait*/ };
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
  // Earliest deliverable packet on (c,d) over both classes, by (arrival, wire_seq).
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
    // Equidistant: alternate CW/CCW so one direction does not take all half-ring traffic.
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
  // Room for a `bytes`-byte packet in deviceID's input buffer (converted to flits)?
  bool HasBuffer(unsigned deviceID, unsigned int bytes) const;
  // Flit sizes (bytes) as mem_fetch uses them; set at init, before the first HasBuffer().
  void set_flit_sizes(unsigned local_flit_bytes, unsigned inter_flit_bytes);
  // Register / query the busy flag of the exact output buffer Pop(dev) reads.
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

  // One local crossbar per chiplet: net[subnet][chiplet].
  std::vector<std::vector<xbar_router*> > net;

  // No ring: move ICP egress packets straight to the destination ICP ingress (no latency).
  void handoff();

  // Device->chiplet owner maps and O(1) lookups, built by configure_chiplets().
  unsigned m_n_chiplets = 1;
  bool     m_chiplets_configured = false;
  std::vector<unsigned> m_dev_chiplet;      // [n_shader+n_mem] -> chiplet id
  std::vector<unsigned> m_local_idx;        // [n_shader+n_mem] -> local node idx in its chiplet
  std::vector<unsigned> m_n_local_cluster;  // [n_chiplets] cluster count per chiplet
  std::vector<unsigned> m_n_local_mem;      // [n_chiplets] mem-sub count per chiplet

  bool m_use_ring = false;                 // ring, or direct handoff when -icnt_use_ring 0
  unsigned m_flit_bytes_local = 0;         // -icnt_flit_size
  unsigned m_flit_bytes_inter = 0;         // -icnt_inter_flit_size
  // Flits a packet of `bytes` bytes occupies in a router input buffer.
  unsigned bytes_to_flits(unsigned bytes) const;
  std::vector<ring_net*> m_ring;           // one ring per subnet (when m_use_ring)
  void handoff_ring();                     // ICP egress<->ring<->ICP ingress

  // [RINGINJ]: egress-busy cycles and failed/successful injects per (subnet, src chiplet).
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
