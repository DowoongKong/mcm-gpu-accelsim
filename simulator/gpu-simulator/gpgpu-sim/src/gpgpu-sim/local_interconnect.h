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
  unsigned ring_link_width;  // flits drained per cycle per link (BW = LW * flit * freq)
  unsigned xbar_port_width;  // intra-chiplet xbar: flits drained per cycle per OUTPUT port
                             // (per-port BW = xbar_port_width * flit_size * icnt_freq).
};

class xbar_router {
 public:
  xbar_router(unsigned router_id, enum Interconnect_type m_type,
              unsigned n_shader, unsigned n_mem,
              const struct inct_config& m_localinct_config,
              unsigned n_icp = 0,
              bool inter_via_ring = false);  // true -> xbar uses LOCAL flit size only
  ~xbar_router();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  void* Pop(unsigned ouput_deviceID);
  void Advance();

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
    }

    void* data;
    unsigned output_deviceID;
    unsigned remaining_flits;
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
  // Fix B: per-output-destination ICP ingress queues (VOQ) + shared-budget flit count
  // + per-output RR turn (alternate VOQ vs local inputs, starvation-free).
  vector<queue<Packet> > m_icp_voq;
  unsigned m_icp_voq_flits;
  vector<unsigned> m_icp_voq_rr;
  unsigned _n_shader, _n_mem, total_nodes;
  unsigned in_buffer_limit, out_buffer_limit;
  vector<unsigned> next_node;  // used for iSLIP arbit
  unsigned next_node_id;       // used for RR arbit
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
  ring_net(unsigned n_nodes, unsigned buf_depth, unsigned hop_latency, unsigned link_width = 1)
      : N(n_nodes), m_buf(buf_depth), m_hop_latency(hop_latency), 
        m_link_width(link_width ? link_width : 1), m_cycle(0) {
    link.assign(N, std::vector<std::queue<RingPkt> >(2));  // [node][dir] 0=cw,1=ccw (bandwidth)
    wire.assign(N, std::vector<std::queue<RingPkt> >(2));  // [node][dir] (wire latency, in-flight)
    eject_buf.assign(N, std::queue<RingPkt>());
    link_flits.assign(N, std::vector<unsigned long long>(2, 0));   // RINGLINK_ (temp)
    link_occ_sum.assign(N, std::vector<unsigned long long>(2, 0)); // RINGLINK_ (temp)
    ringlat_src_sum.assign(N, 0);   // RINGSRC_ (temp)
    ringlat_src_cnt.assign(N, 0);   // RINGSRC_ (temp)
    bkt_bw_sum.assign(N, 0);        // BUCKET_ (temp)
    bkt_ej_sum.assign(N, 0);        // BUCKET_ (temp)
  }
  // Inject at node src toward dst. false (no-op) if the outgoing link buffer is full.
  bool inject(unsigned src, unsigned dst, unsigned final_dev, void* data, unsigned flits) {
    unsigned dir = dir_of(src, dst);
    if (link[src][dir].size() >= m_buf) return false;
    RingPkt p; p.data = data; p.final_dev = final_dev; p.dst = dst;
    p.dir = dir; p.flits = flits; p.remaining = flits; p.arrival = 0;
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
    ++link_occ_samples;   // RINGLINK_ (temp)
    // Phase A: drain up to m_link_width flits per link per cycle, spanning packets
    // (so small packets can pass multiple-per-cycle -> high link bandwidth).
    // BW(per link/dir) = m_link_width * inter_flit_size * freq.
    for (unsigned c = 0; c < N; ++c)
      for (unsigned d = 0; d < 2; ++d) {
        link_occ_sum[c][d] += link[c][d].size();   // RINGLINK_ (temp): queue occupancy
        unsigned budget = m_link_width;
        while (budget > 0 && !link[c][d].empty()) {
          RingPkt& p = link[c][d].front();
          unsigned take = (p.remaining < budget) ? p.remaining : budget;
          link_flits[c][d] += take;   // RINGLINK_ (temp): flits transmitted on this link
          p.remaining -= take;
          budget      -= take;
          if (p.remaining == 0) {              // finished on this link -> wire (latency) stage
            p.t_bw_sum += (m_cycle - p.enter_link);  /*BUCKET_*/ // time spent in link stage
            p.enter_wire = m_cycle;                  /*BUCKET_*/
            p.arrival = m_cycle + m_hop_latency;
            wire[c][d].push(p);
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
        while (!wire[c][d].empty() && wire[c][d].front().arrival <= m_cycle) {
          RingPkt p = wire[c][d].front();
          unsigned next = neighbor(c, d);
          // BUCKET_ (temp): wait beyond the legit wire (hop) latency at this hop.
          unsigned long long ejw = (m_cycle - p.enter_wire > m_hop_latency)
                                   ? (m_cycle - p.enter_wire - m_hop_latency) : 0;
          if (next == p.dst) {                          // arrived at destination chiplet
            if (eject_buf[next].size() >= m_buf) break; // eject full -> wait
            unsigned long long rt = m_cycle - p.inject_cycle;  /*RINGLAT_*/
            ringlat_sum += rt; ringlat_cnt++;                   /*RINGLAT_*/
            if (rt > ringlat_max) ringlat_max = rt;             /*RINGLAT_*/
            p.t_ej_sum += ejw;                                  /*BUCKET_*/
            if (p.src < ringlat_src_sum.size()) {               /*RINGSRC_*/
              ringlat_src_sum[p.src] += rt; ringlat_src_cnt[p.src]++;
              bkt_bw_sum[p.src] += p.t_bw_sum;                  /*BUCKET_*/
              bkt_ej_sum[p.src] += p.t_ej_sum;                  /*BUCKET_*/
            }
            eject_buf[next].push(p); wire[c][d].pop();
          } else {                                      // forward to next hop's link
            if (link[next][d].size() >= m_buf) break;   // downstream link full -> wait
            p.t_ej_sum += ejw;                          /*BUCKET_*/
            p.enter_link = m_cycle;                     /*BUCKET_*/
            p.remaining = p.flits;                      // re-transmit (bandwidth) on next link
            link[next][d].push(p); wire[c][d].pop();
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
        if (!link[c][d].empty() || !wire[c][d].empty()) return true;
    }
    return false;
  }
 public:
  // RINGLAT_ (temp): ring traversal-time stats (inject -> eject), inter-chiplet only.
  unsigned long long ringlat_sum = 0, ringlat_cnt = 0, ringlat_max = 0;
  // RINGLINK_ (temp): per-(node,dir) link load. link_flits = flits transmitted (throughput),
  // link_occ_sum/link_occ_samples = avg queue occupancy. Shows tie-break CW/CCW imbalance.
  std::vector<std::vector<unsigned long long> > link_flits;
  std::vector<std::vector<unsigned long long> > link_occ_sum;
  unsigned long long link_occ_samples = 0;
  unsigned ring_N() const { return N; }
  // RINGSRC_ (temp): per-SOURCE-chiplet ring residency (inject->eject). Isolates
  // whether a source's ring time is high (e.g. eject-order unfairness at the hub).
  std::vector<unsigned long long> ringlat_src_sum;
  std::vector<unsigned long long> ringlat_src_cnt;
  // BUCKET_ (temp): per-source ring-residency breakdown. t_wire = total - bw - ejwait.
  //   bw     = time in link[] bandwidth stage (serialization/queue for the link)
  //   ejwait = time in wire[] AFTER arrival passed but blocked from eject/forward
  //            (this is the artifact shadow: ICP-ingress/downstream backpressure).
  std::vector<unsigned long long> bkt_bw_sum;
  std::vector<unsigned long long> bkt_ej_sum;
 private:
  struct RingPkt { void* data; unsigned final_dev, dst, dir, flits, remaining, src;
                   unsigned long long arrival;
                   unsigned long long inject_cycle; /*RINGLAT_*/
                   unsigned long long enter_link, enter_wire; /*BUCKET_*/
                   unsigned long long t_bw_sum, t_ej_sum;     /*BUCKET_*/ };
  unsigned N, m_buf, m_hop_latency;
  unsigned m_link_width;  // flits/cycle per link
  unsigned long long m_cycle;
  mutable unsigned m_tie_rr = 0;  // round-robin cursor for equidistant direction split
  std::vector<std::vector<std::queue<RingPkt> > > link;  // bandwidth stage [node][dir]
  std::vector<std::vector<std::queue<RingPkt> > > wire;  // latency stage   [node][dir]
  std::vector<std::queue<RingPkt> >               eject_buf;
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
