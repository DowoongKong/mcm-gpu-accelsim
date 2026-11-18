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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include "local_interconnect.h"
#include "mem_fetch.h"

// Class name for the RING* rows (RingPkt::cls: 0 = data, 1 = page-table walk).
static const char* ring_cls_name(unsigned cls) { return cls ? "ptw" : "data"; }

// ACCELSIM_DIAG: current gpu cycle, used to stamp a packet's arrival at its xbar output buffer.
unsigned long long g_icnt_now = 0;

// ACCELSIM_PHASE_TIMING: ICNT wall time, split into xbar Advance() and ring/handoff.
extern const bool g_phase_timing_on;
extern unsigned long long g_icnt_local_ns, g_icnt_handoff_ns;
extern unsigned long long g_icnt_local_calls, g_icnt_handoff_calls;
static inline unsigned long long icnt_split_now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

// ACCELSIM_DIAG: non-PTW reply xbar latency per source mem chiplet.
unsigned long long g_normrep_transit_sum[8] = {0};
unsigned long long g_normrep_lasthop_sum[8] = {0};
unsigned long long g_normrep_cnt[8]         = {0};
// ACCELSIM_DIAG: non-PTW reply transit per [source mem chiplet][dest cluster chiplet].
unsigned long long g_normrep_mtx_transit[8][8] = {{0}};
unsigned long long g_normrep_mtx_cnt[8][8]     = {{0}};

xbar_router::xbar_router(unsigned router_id, enum Interconnect_type m_type,
                         unsigned n_shader, unsigned n_mem,
                         const struct inct_config& m_localinct_config,
                         unsigned n_icp, bool inter_via_ring, bool both_dirs) {
  m_id = router_id;
  router_type = m_type;
  _n_mem = n_mem;
  _n_shader = n_shader;
  _n_icp = n_icp;
  m_inter_via_ring = inter_via_ring;
  total_nodes = n_shader + n_mem + n_icp;  // clusters, mems, then the ICP
  verbose = m_localinct_config.verbose;
  grant_cycles = m_localinct_config.grant_cycles;
  grant_cycles_count = m_localinct_config.grant_cycles;
  in_buffers.resize(total_nodes);
  out_buffers.resize(total_nodes);
  m_out_mirror.assign(total_nodes, nullptr);   // busy flags
  m_icp_voq.resize(total_nodes);          // ICP ingress VOQ, one per output
  m_icp_voq_flits = 0;
  m_icp_voq_rr.assign(total_nodes, 0);
  next_node.resize(total_nodes, 0);
  in_buffer_limit = m_localinct_config.in_buffer_limit;
  out_buffer_limit = m_localinct_config.out_buffer_limit;
  arbit_type = m_localinct_config.arbiter_algo;
  m_port_width = m_localinct_config.xbar_port_width ? m_localinct_config.xbar_port_width : 1;
  next_node_id = 0;
  if (m_type == REQ_NET) {
    active_in_buffers = n_shader;
    active_out_buffers = n_mem;
  } else if (m_type == REPLY_NET) {
    active_in_buffers = n_mem;
    active_out_buffers = n_shader;
  }

  // Candidate outputs (see m_out_cand); with one subnet every output is a candidate.
  m_out_cand.clear();
  if (both_dirs) {
    for (unsigned o = 0; o < total_nodes; ++o) m_out_cand.push_back(o);
  } else if (router_type == REQ_NET) {
    for (unsigned o = _n_shader; o < _n_shader + _n_mem; ++o) m_out_cand.push_back(o);
    if (_n_icp) m_out_cand.push_back(icp_index());
  } else {
    for (unsigned o = 0; o < _n_shader; ++o) m_out_cand.push_back(o);
    if (_n_icp) m_out_cand.push_back(icp_index());
  }
  // Pending-input counts, one slot per possible destination index.
  m_out_pending.assign(total_nodes, 0u);
  // Seen-destination stamps for the conflict count, one slot per possible destination index.
  m_seen_stamp.assign(total_nodes, 0ull);
  m_seen_cur = 0ull;

  cycles = 0;
  conflicts = 0;
  out_buffer_full = 0;
  in_buffer_full = 0;
  out_buffer_util = 0;
  in_buffer_util = 0;
  packets_num = 0;
  conflicts_util = 0;
  cycles_util = 0;
  reqs_util = 0;
}

xbar_router::~xbar_router() {}

void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID,
                       void* data, unsigned int size) {
  assert(input_deviceID < total_nodes);

  mem_fetch* mf = static_cast<mem_fetch*>(data);
  unsigned flits = 1;  // same fallback as flits_of()
  // With the ring, the xbar uses the local flit size; the ring models inter-chiplet bandwidth.
  bool is_local = m_inter_via_ring ? true : mf->is_local_access();
  if (mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_REQUEST
      || mf->get_type() == TLB_FILL_ACC)
    flits = mf->get_num_flits(true, is_local);
  else if (mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK
      || mf->get_type() == TLB_FILL_REPLY)
    flits = mf->get_num_flits(false, is_local);

  // The grant loop skips non-candidate outputs; debug-only check (linear scan, hot path).
#ifdef XBAR_CAND_CHECK
  assert(is_out_candidate(output_deviceID) &&
         "packet addressed to a non-candidate output");
#endif
  in_buffers[input_deviceID].push(Packet(data, output_deviceID, flits));
  // Shared statistic bumped from every source node during the CORE phase.
#pragma omp atomic
  packets_num++;
}

void* xbar_router::Pop(unsigned ouput_deviceID) {
  assert(ouput_deviceID < total_nodes);
  void* data = NULL;

  if (!out_buffers[ouput_deviceID].empty()) {
    data = out_buffers[ouput_deviceID].front().data;
    out_buffers[ouput_deviceID].pop();
  }

  return data;
}

unsigned xbar_router::queue_flits(const flit_queue& q) const {
  // running total maintained by flit_queue
#ifdef XBAR_FLITS_CHECK
  assert(q.flits == q.recount() && "flit_queue invariant broken");
#endif
  return q.flits;
}

bool xbar_router::Has_Buffer_In(unsigned input_deviceID, unsigned flits, 
                                bool update_counter) {
  assert(input_deviceID < total_nodes);

  unsigned occ = queue_flits(in_buffers[input_deviceID]);

  bool has_buffer = (occ + flits <= in_buffer_limit);

  if (update_counter && !has_buffer) {
    // Atomic: called from every SM thread (the other xbar counters are router-private).
#pragma omp atomic
    in_buffer_full++;
  }

  return has_buffer;
}

bool xbar_router::Has_Buffer_Out(unsigned output_deviceID, unsigned flits) {
  assert(output_deviceID < total_nodes);

  // Occupancy in flits; the grant loop probes with flits == 1, so it can overshoot by one packet.
  unsigned occ = queue_flits(out_buffers[output_deviceID]);

  return (occ + flits <= out_buffer_limit);
}

void* xbar_router::TopOut(unsigned out) const {
  assert(out < total_nodes);
  return out_buffers[out].empty() ? nullptr : out_buffers[out].front().data;
}

bool xbar_router::HasOut(unsigned out) const {
  assert(out < total_nodes);
  return !out_buffers[out].empty();
}

unsigned xbar_router::flits_of(void* data) const {
  mem_fetch* mf = static_cast<mem_fetch*>(data);
  bool is_local = m_inter_via_ring ? true : mf->is_local_access();
  if (mf->get_type()==READ_REQUEST || mf->get_type()==WRITE_REQUEST
      || mf->get_type()==TLB_FILL_ACC)
    return mf->get_num_flits(true,  is_local);
  else if (mf->get_type()==READ_REPLY || mf->get_type()==WRITE_ACK
      || mf->get_type() == TLB_FILL_REPLY)
    return mf->get_num_flits(false, is_local);
  return 1;
}

void xbar_router::Advance() {
  if (arbit_type == NAIVE_RR)
    RR_Advance();
  else if (arbit_type == iSLIP)
    iSLIP_Advance();
  else
    assert(0);
}

void xbar_router::RR_Advance() {
  bool active = false;
  // issued[out] = grants to output `out` in the cycle, up to m_port_width.
  vector<unsigned> issued(total_nodes, 0);
  unsigned conflict_sub = 0;
  unsigned reqs = 0;

  for (unsigned i = 0; i < total_nodes; ++i) {
    unsigned node_id = (i + next_node_id) % total_nodes;

    if (!in_buffers[node_id].empty()) {
      active = true;
      Packet _packet = in_buffers[node_id].front();
      // ensure that the outbuffer has space and not issued before in this cycle
      if (Has_Buffer_Out(_packet.output_deviceID, 1)) {
        if (issued[_packet.output_deviceID] < m_port_width) {
          // Diagnostics: stamp the packet's out_buffer arrival (see iSLIP_Advance).
          mem_fetch* _rmf = static_cast<mem_fetch*>(_packet.data);
          if (g_accelsim_diag && _rmf) _rmf->set_icnt_receive_time((unsigned)g_icnt_now);  // diagnostics only
          out_buffers[_packet.output_deviceID].push(_packet);
          mark_out(_packet.output_deviceID);   // busy flag
          in_buffers[node_id].pop();
          issued[_packet.output_deviceID]++;
          reqs++;
        } else
          conflict_sub++;
      } else {
        {  // router-private counter (the atomic is not needed)
        #pragma omp atomic
          out_buffer_full++;
        }

        if (issued[_packet.output_deviceID]) conflict_sub++;
      }
    }
  }
  next_node_id = next_node_id + 1;
  next_node_id = (next_node_id % total_nodes);

  {  // router-private counter (the atomic is not needed)
  #pragma omp atomic
    conflicts += conflict_sub;
  }
  if (active) {
    {  // router-private counter (the atomic is not needed)
    #pragma omp atomic
      conflicts_util += conflict_sub;
    }
    {  // router-private counter (the atomic is not needed)
    #pragma omp atomic
      cycles_util++;
    }
    {  // router-private counter (the atomic is not needed)
    #pragma omp atomic
      reqs_util += reqs;
    }
  }

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  // collect some stats about buffer util (over all nodes: any node can hold packets)
  unsigned long long _in_u = 0, _out_u = 0;
  for (unsigned i = 0; i < total_nodes; ++i) {
    _in_u  += in_buffers[i].size();
    _out_u += out_buffers[i].size();
  }
  in_buffer_util  += _in_u;
  out_buffer_util += _out_u;

  cycles++;   // router-private counter
}

// iSLIP algorithm
// McKeown, Nick. "The iSLIP scheduling algorithm for input-queued switches."
// IEEE/ACM transactions on networking 2 (1999): 188-201.
// https://www.cs.rutgers.edu/~sn624/552-F18/papers/islip.pdf
void xbar_router::iSLIP_Advance() {
  bool active = false;

  unsigned conflict_sub = 0;
  unsigned reqs = 0;

  // calcaulte how many conflicts are there for stats
  // A later input naming an already-seen destination is a conflict (per-cycle stamp, O(1)).
  ++m_seen_cur;
  // Rebuild the pending-input counts in the same pass.
  std::fill(m_out_pending.begin(), m_out_pending.end(), 0u);
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) {
      const unsigned _tgt = in_buffers[i].front().output_deviceID;
      m_out_pending[_tgt]++;
      if (m_seen_stamp[_tgt] == m_seen_cur)
        conflict_sub++;
      else
        m_seen_stamp[_tgt] = m_seen_cur;
      active = true;
    }
  }

  // Router-private counters (one thread per router): no atomic. Only in_buffer_full is shared.
  conflicts += conflict_sub;
  if (active) {
    conflicts_util += conflict_sub;
    cycles_util++;
  }
  // Idle router (no input packet, empty ICP VOQ): nothing can be granted, so skip the scan and
  // only count full output buffers. Exact: the iSLIP pointers move only on a grant.
  if (!active && m_icp_voq_flits == 0) {
    unsigned obf = 0;
    // Non-candidate outputs stay empty, so they can never be counted.
    for (unsigned oi = 0; oi < m_out_cand.size(); ++oi)
      if (!Has_Buffer_Out(m_out_cand[oi], 1)) obf++;
    if (obf) out_buffer_full += obf;   // router-private counter
  } else {
  // do iSLIP; each output grants up to m_port_width flits per cycle, across packets and inputs.
  for (unsigned oi = 0; oi < m_out_cand.size(); ++oi) {
   const unsigned i = m_out_cand[oi];   // candidate outputs only
   for (unsigned w = 0; w < m_port_width; ++w) {
    bool _did_grant = false;
    if (Has_Buffer_Out(i, 1)) {
      // Nothing addressed here: stop. Exact, since the scan would grant and change nothing.
      if (m_out_pending[i] == 0 && m_icp_voq[i].empty()) {
#ifdef XBAR_CAND_CHECK   // no input front may be addressed to i
        for (unsigned _j = 0; _j < total_nodes; ++_j)
          assert(in_buffers[_j].empty() || in_buffers[_j].front().output_deviceID != i);
#endif
        break;
      }
      // Alternate ICP VOQ[i] and local inputs (no starvation); pass 1 falls back to the other.
      bool _voq_first = (m_icp_voq_rr[i] & 1u);
      for (int _pass = 0; _pass < 2 && !_did_grant; ++_pass) {
        bool _use_voq = (_pass == 0) ? _voq_first : !_voq_first;
        if (_use_voq) {
          // (A) serve one flit of ICP VOQ[i] front (per-flit -> data BW preserved)
          if (!m_icp_voq[i].empty()) {
            Packet &_vp = m_icp_voq[i].front();
            _vp.remaining_flits--;
            m_icp_voq_flits--;                          // shared budget
            if (_vp.remaining_flits == 0) {
              mem_fetch* _rmf = static_cast<mem_fetch*>(_vp.data);
              if (g_accelsim_diag && _rmf) _rmf->set_icnt_receive_time((unsigned)g_icnt_now);  // diagnostics only
              Packet _done = _vp;
              _done.remaining_flits = _done.orig_flits;  // out-buffer occupancy
              out_buffers[i].push(_done);                // = full packet size
              mark_out(i);                               // busy flag
              m_icp_voq[i].pop();
            }
            reqs++;
            _did_grant = true;
          }
        } else {
          // (B) serve a local input (iSLIP scan over in_buffers)
          for (unsigned j = 0; j < total_nodes; ++j) {
            unsigned node_id = (j + next_node[i]) % total_nodes;
            if (!in_buffers[node_id].empty()) {
              const Packet &_packet = in_buffers[node_id].front();
              if (_packet.output_deviceID == i) {
                in_buffers[node_id].dec_front_flits();   // keeps the running total
                if (_packet.remaining_flits == 0) {
                  mem_fetch* _rmf = static_cast<mem_fetch*>(_packet.data);
                  if (g_accelsim_diag && _rmf) _rmf->set_icnt_receive_time((unsigned)g_icnt_now);  // diagnostics only
                  Packet _done = _packet;
                  _done.remaining_flits = _done.orig_flits;  // out-buffer occupancy
                  out_buffers[_packet.output_deviceID].push(_done);  // = full packet size
                  mark_out(_packet.output_deviceID);                 // busy flag
                  in_buffers[node_id].pop();
                  // Front changed: move the pending count from i to the new front's output.
                  m_out_pending[i]--;
                  if (!in_buffers[node_id].empty())
                    m_out_pending[in_buffers[node_id].front().output_deviceID]++;
                }
                if (grant_cycles_count == 1)
                  next_node[i] = (++node_id % total_nodes);
                reqs++;
                _did_grant = true;
                break;
              }
            }
          }
        }
      }
      if (_did_grant) m_icp_voq_rr[i]++;   // advance RR turn on a grant
    } else
      out_buffer_full++;      // router-private counter
    if (!_did_grant) break;   // out buffer full or no packet for this output -> stop
   }
  }

  }  // end of the idle-router skip

  if (active) reqs_util += reqs;   // router-private counter

  if (verbose)
    printf("%d : cycle %llu : grant_cycles = %d\n", m_id, cycles, grant_cycles);

  if (active && grant_cycles_count == 1)
    grant_cycles_count = grant_cycles;
  else if (active)
    grant_cycles_count--;

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  // collect some stats about buffer util (over all nodes: any node can hold packets)
  unsigned long long _in_u = 0, _out_u = 0;
  for (unsigned i = 0; i < total_nodes; ++i) {
    _in_u  += in_buffers[i].size();
    _out_u += out_buffers[i].size();
  }
  in_buffer_util  += _in_u;
  out_buffer_util += _out_u;

  cycles++;   // router-private counter
}

bool xbar_router::Busy() const {
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) return true;

    if (!out_buffers[i].empty()) return true;

    if (!m_icp_voq[i].empty()) return true;   // pending ICP ingress VOQ packets
  }
  return false;
}

////////////////////////////////////////////////////
/////////////LocalInterconnect/////////////////////

// Unused (GetFlitSize() only); real sizes: -icnt_flit_size, -icnt_inter_flit_size.
#define LOCAL_INCT_FLIT_SIZE 40

LocalInterconnect* LocalInterconnect::New(
    const struct inct_config& m_localinct_config) {
  LocalInterconnect* icnt_interface = new LocalInterconnect(m_localinct_config);

  return icnt_interface;
}

LocalInterconnect::LocalInterconnect(
    const struct inct_config& m_localinct_config)
    : m_inct_config(m_localinct_config) {
  n_shader = 0;
  n_mem = 0;
  n_subnets = m_localinct_config.subnets;
}

LocalInterconnect::~LocalInterconnect() {
  for (unsigned s = 0; s < net.size(); ++s)
    for (unsigned ch = 0; ch < net[s].size(); ++ch)
      delete net[s][ch];
  for (unsigned s = 0; s < m_ring.size(); ++s) delete m_ring[s];
}

void LocalInterconnect::CreateInterconnect(unsigned m_n_shader, unsigned m_n_mem) {
  n_shader = m_n_shader;  n_mem = m_n_mem;
  // Validate the chiplet maps here: n_shader/n_mem are still 0 in configure_chiplets().
  assert(m_chiplets_configured && "configure_chiplets must run before CreateInterconnect");
  assert(m_dev_chiplet.size() == n_shader + n_mem && "chiplet map size mismatch");

  m_use_ring = (m_inct_config.use_ring != 0) && (m_n_chiplets > 1);
  const unsigned n_icp = (m_n_chiplets > 1) ? 1u : 0u;
  net.resize(n_subnets);
  for (unsigned s = 0; s < n_subnets; ++s) {
    net[s].resize(m_n_chiplets, nullptr);
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch) {
      net[s][ch] = new xbar_router(ch, static_cast<Interconnect_type>(s),
                                   m_n_local_cluster[ch], m_n_local_mem[ch],
                                   m_inct_config, n_icp, m_use_ring,  // inter_via_ring
                                   /*both_dirs=*/(n_subnets == 1));
    }
  }
  if (m_use_ring) {
    m_ring.assign(n_subnets, nullptr);
    // A negative -icnt_ptw_ring_hop_latency means same as the data class.
    const int      ptw_cfg = m_inct_config.ring_hop_latency_ptw;
    const unsigned ptw_hop = (ptw_cfg < 0) ? m_inct_config.ring_hop_latency
                                           : (unsigned)ptw_cfg;
    printf("[ICNT] ring hop latency: data=%u ptw=%u cycles%s\n",
           m_inct_config.ring_hop_latency, ptw_hop,
           (ptw_cfg < 0) ? " (ptw inherited)" : "");
    for (unsigned s = 0; s < n_subnets; ++s)
      m_ring[s] = new ring_net(m_n_chiplets, m_inct_config.ring_buf,
                               m_inct_config.ring_hop_latency,
                               ptw_hop,
                               m_inct_config.ring_link_width);
    // [RINGINJ] diagnostics: [subnet][chiplet] source-side egress accounting.
    m_rinj_egress_busy.assign(n_subnets, std::vector<unsigned long long>(m_n_chiplets, 0));
    m_rinj_stall.assign(n_subnets, std::vector<unsigned long long>(m_n_chiplets, 0));
    m_rinj_ok.assign(n_subnets, std::vector<unsigned long long>(m_n_chiplets, 0));
  }
}

void LocalInterconnect::Init() {
  // empty
  // there is nothing to do
}

void LocalInterconnect::Push(unsigned in, unsigned out, void* data, unsigned int size) {
  unsigned subnet = (n_subnets == 1) ? 0
                  : (in < n_shader ? REQ_NET : REPLY_NET);
  unsigned src_ch = m_dev_chiplet[in];
  unsigned dst_ch = m_dev_chiplet[out];

  if (src_ch == dst_ch) {                                      // LOCAL
    // The admitted packet must fit by the flits it really occupies.
    assert(net[subnet][src_ch]->Has_Buffer_In(m_local_idx[in], net[subnet][src_ch]->flits_of(data)));
    net[subnet][src_ch]->Push(m_local_idx[in], m_local_idx[out], data, size);
  } else {                                                     // REMOTE
    // The admitted packet must fit by the flits it really occupies.
    assert(net[subnet][src_ch]->Has_Buffer_In(m_local_idx[in], net[subnet][src_ch]->flits_of(data)));
    net[subnet][src_ch]->Push(m_local_idx[in], net[subnet][src_ch]->icp_index(), data, size);
  }
}

// Same subnet / chiplet / local-index rule as Pop() (the consumer side).
void LocalInterconnect::SetOutMirror(unsigned out, unsigned char *slot) {
  unsigned subnet = (out < n_shader) ? REPLY_NET : REQ_NET;
  if (n_subnets == 1) subnet = 0;
  net[subnet][m_dev_chiplet[out]]->set_out_mirror(m_local_idx[out], slot);
}
bool LocalInterconnect::OutEmpty(unsigned out) const {
  unsigned subnet = (out < n_shader) ? REPLY_NET : REQ_NET;
  if (n_subnets == 1) subnet = 0;
  return net[subnet][m_dev_chiplet[out]]->out_empty(m_local_idx[out]);
}

void* LocalInterconnect::Pop(unsigned out) {
  // Subnet chosen by the consumer side.
  unsigned subnet = (out < n_shader) ? REPLY_NET : REQ_NET;
  if (n_subnets == 1) subnet = 0;
  unsigned ch = m_dev_chiplet[out];
  return net[subnet][ch]->Pop(m_local_idx[out]);
}

void LocalInterconnect::Advance() {
  // Routers advance in parallel (each has private state); rings and handoff stay serial.
  const unsigned long long xbar_advance_wall_start = g_phase_timing_on ? icnt_split_now_ns() : 0;
#pragma omp parallel for schedule(runtime) collapse(2)
  for (unsigned s = 0; s < n_subnets; ++s)
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch)
      net[s][ch]->Advance();
  if (g_phase_timing_on) {
    g_icnt_local_ns += icnt_split_now_ns() - xbar_advance_wall_start;
    g_icnt_local_calls++;
  }

  const unsigned long long handoff_wall_start = g_phase_timing_on ? icnt_split_now_ns() : 0;
  if (m_n_chiplets > 1) {
    if (m_use_ring) {
      for (unsigned s = 0; s < n_subnets; ++s) m_ring[s]->advance();
      handoff_ring();
    } else {
      handoff();              // direct handoff, no ring (-icnt_use_ring 0)
    }
  }
  if (g_phase_timing_on) {
    g_icnt_handoff_ns += icnt_split_now_ns() - handoff_wall_start;
    g_icnt_handoff_calls++;
  }
}

void LocalInterconnect::handoff() {
  for (unsigned s = 0; s < n_subnets; ++s) {
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch) {
      xbar_router* src = net[s][ch];
      unsigned egress = src->icp_index();
      // Drain the ICP egress to the destination chiplets (several packets per Advance).
      while (src->HasOut(egress)) {
        void* data = src->TopOut(egress);
        mem_fetch* mf = static_cast<mem_fetch*>(data);
        // Reconstruct the final GLOBAL device id (per subnet).
        unsigned final_dev = (s == REQ_NET)
            ? (n_shader + mf->get_sub_partition_id())   // mem2device
            : (mf->get_tpc());                          // cluster deviceID
        unsigned dch = m_dev_chiplet[final_dev];
        xbar_router* dst = net[s][dch];
        unsigned ingress = dst->icp_index();
        unsigned f = dst->flits_of(data);
        if (!dst->Has_Buffer_In(ingress, f)) break;     // backpressure: retry next Advance
        (void) src->Pop(egress);                        // remove it
        dst->Push(ingress, m_local_idx[final_dev], data, 0);  // size ignored (flits from mf)
      }
    }
  }
}

void LocalInterconnect::handoff_ring() {
  if (g_accelsim_diag) m_rinj_advances++;   // [RINGINJ]
  for (unsigned s = 0; s < n_subnets; ++s) {
    ring_net* ring = m_ring[s];

    // (1) inject: each chiplet's ICP egress -> ring
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch) {
      xbar_router* src = net[s][ch];
      unsigned egress = src->icp_index();
      bool had_pending = src->HasOut(egress);   // [RINGINJ]
      bool stalled = false;                     // [RINGINJ]
      while (src->HasOut(egress)) {
        void* data = src->TopOut(egress);
        mem_fetch* mf = static_cast<mem_fetch*>(data);
        unsigned final_dev = (s == REQ_NET)
            ? (n_shader + mf->get_sub_partition_id())   // mem2device
            : (mf->get_tpc());                          // cluster deviceID
        unsigned dch   = m_dev_chiplet[final_dev];
        // Ring link width = inter flit size (NOT the local xbar's local size).
        unsigned flits = mf->get_num_flits(s == REQ_NET, /*is_local=*/false);
        // Page-table-walk packets (TLB_FILL_ACC / TLB_FILL_REPLY) use ring class 1.
        const bool is_ptw = (mf->get_type() == TLB_FILL_ACC ||
                             mf->get_type() == TLB_FILL_REPLY);
        if (!ring->inject(ch, dch, final_dev, data, flits, is_ptw ? 1u : 0u)) {
          stalled = true; break;   // link full
        }
        (void) src->Pop(egress);
        if (g_accelsim_diag) m_rinj_ok[s][ch]++;  // [RINGINJ]
      }
      // [RINGINJ]: egress had work this cycle; stalled = blocked by ring-link-full.
      if (g_accelsim_diag && had_pending) {
        m_rinj_egress_busy[s][ch]++;
        if (stalled) m_rinj_stall[s][ch]++;
      }
    }

    // (2) eject: ring -> dst chiplet's ICP ingress VOQ (per-output queues, then iSLIP per flit).
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch) {
      while (ring->has_eject(ch)) {
        void* data        = ring->top_eject(ch);
        unsigned final_dev = ring->eject_final_dev(ch);   // chiplet(final_dev) == ch
        xbar_router* dst  = net[s][ch];
        unsigned out      = m_local_idx[final_dev];       // destination output port
        unsigned f        = dst->flits_of(data);
        if (!dst->has_icp_voq_room(f)) break;             // shared VOQ budget full -> retry
        ring->pop_eject(ch);
        dst->push_icp_voq(out, data, f);                  // enqueue to that output's VOQ
      }
    }
  }
}

bool LocalInterconnect::Busy() const {
  for (unsigned s = 0; s < n_subnets; ++s)
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch)
      if (net[s][ch]->Busy()) return true;   // xbar Busy() covers the ICP buffers too
  for (unsigned s = 0; s < m_ring.size(); ++s)
    if (m_ring[s] && m_ring[s]->busy()) return true;
  return false;
}

bool LocalInterconnect::HasBuffer(unsigned deviceID, unsigned int bytes) const {
  unsigned subnet = (n_subnets > 1 && deviceID >= n_shader) ? REPLY_NET : REQ_NET;
  unsigned ch     = m_dev_chiplet[deviceID];
  return net[subnet][ch]->Has_Buffer_In(m_local_idx[deviceID], bytes_to_flits(bytes), true);
}

void LocalInterconnect::set_flit_sizes(unsigned local_flit_bytes, unsigned inter_flit_bytes) {
  assert(local_flit_bytes > 0 && inter_flit_bytes > 0);
  m_flit_bytes_local = local_flit_bytes;
  m_flit_bytes_inter = inter_flit_bytes;
}

// Same flit rule as xbar_router::Push(). Exact with the ring or one chiplet; otherwise the
// destination is unknown here, so use the smaller flit size (can only over-count).
unsigned LocalInterconnect::bytes_to_flits(unsigned bytes) const {
  assert(m_flit_bytes_local > 0 && "set_flit_sizes() must run before HasBuffer()");
  const unsigned flit = (m_use_ring || m_n_chiplets == 1)
                            ? m_flit_bytes_local
                            : std::min(m_flit_bytes_local, m_flit_bytes_inter);
  return (bytes + flit - 1) / flit;
}

void LocalInterconnect::DisplayStats() const {
  // All routers advance every cycle, so chiplet 0's cycle count serves for *_per_cycle.
  for (unsigned s = 0; s < n_subnets; ++s) {
    const char* tag = (s == REQ_NET) ? "Req" : "Reply";
    unsigned long long packets=0, conflicts=0, conflicts_util=0, reqs_util=0;
    unsigned long long in_buffer_full=0, in_buffer_util=0;
    unsigned long long out_buffer_full=0, out_buffer_util=0;
    unsigned long long active_in=0, active_out=0, cyc_util_sum=0;
    for (unsigned ch = 0; ch < net[s].size(); ++ch) {
      xbar_router* r = net[s][ch];
      packets+=r->packets_num; conflicts+=r->conflicts;
      conflicts_util+=r->conflicts_util; reqs_util+=r->reqs_util;
      in_buffer_full+=r->in_buffer_full; in_buffer_util+=r->in_buffer_util;
      out_buffer_full+=r->out_buffer_full; out_buffer_util+=r->out_buffer_util;
      active_in+=r->active_in_buffers; active_out+=r->active_out_buffers;
      cyc_util_sum+=r->cycles_util;
    }
    unsigned long long cyc = net[s][0]->cycles;
    printf("%s_Network_injected_packets_num = %lld\n", tag, packets);
    printf("%s_Network_cycles = %lld\n", tag, cyc);
    printf("%s_Network_injected_packets_per_cycle = %12.4f \n", tag, (float)packets/cyc);
    printf("%s_Network_conflicts_per_cycle = %12.4f\n", tag, (float)conflicts/cyc);
    printf("%s_Network_conflicts_per_cycle_util = %12.4f\n", tag, (float)conflicts_util/cyc_util_sum);
    printf("%s_Bank_Level_Parallism = %12.4f\n", tag, (float)reqs_util/cyc_util_sum);
    printf("%s_Network_in_buffer_full_per_cycle = %12.4f\n", tag, (float)in_buffer_full/cyc);
    printf("%s_Network_in_buffer_avg_util = %12.4f\n", tag, (float)in_buffer_util/cyc/active_in);
    printf("%s_Network_out_buffer_full_per_cycle = %12.4f\n", tag, (float)out_buffer_full/cyc);
    printf("%s_Network_out_buffer_avg_util = %12.4f\n", tag, (float)out_buffer_util/cyc/active_out);
    printf("\n");
  }
  // Ring diagnostic report, only under ACCELSIM_DIAG (the counters are gated too).
  if (m_use_ring && g_accelsim_diag) {
    for (unsigned s = 0; s < m_ring.size(); ++s) {
      if (!m_ring[s]) continue;
      const char* tag = (s == REQ_NET) ? "Req" : "Reply";
      // Combined row, then one row per packet class.
      unsigned long long cnt = 0, sum = 0, mx = 0;
      for (unsigned k = 0; k < ring_net::RING_CLS; ++k) {
        cnt += m_ring[s]->ringlat_cnt[k];
        sum += m_ring[s]->ringlat_sum[k];
        if (m_ring[s]->ringlat_max[k] > mx) mx = m_ring[s]->ringlat_max[k];
      }
      printf("[RINGLAT] %s_ring pkts=%llu avg_ring_cycles=%.2f max_ring_cycles=%llu\n",
             tag, cnt, cnt ? (double)sum / cnt : 0.0, mx);
      for (unsigned k = 0; k < ring_net::RING_CLS; ++k) {
        unsigned long long c = m_ring[s]->ringlat_cnt[k];
        printf("[RINGLAT] %s_ring class=%s pkts=%llu avg_ring_cycles=%.2f "
               "max_ring_cycles=%llu\n",
               tag, ring_cls_name(k), c,
               c ? (double)m_ring[s]->ringlat_sum[k] / c : 0.0,
               m_ring[s]->ringlat_max[k]);
      }
    }
    // [RINGSRC]: ring time per source chiplet, split into bw / wire / ejwait.
    for (unsigned s = 0; s < m_ring.size(); ++s) {
      const char* tag = (s == REQ_NET) ? "Req" : "Reply";
      unsigned N = m_ring[s]->ring_N();
      for (unsigned c = 0; c < N; ++c) {
        // Combined row, then one row per packet class (wire = total - bw - ejwait).
        unsigned long long n = 0, rs = 0, bs = 0, es = 0;
        for (unsigned k = 0; k < ring_net::RING_CLS; ++k) {
          n  += m_ring[s]->ringlat_src_cnt[k][c];
          rs += m_ring[s]->ringlat_src_sum[k][c];
          bs += m_ring[s]->bkt_bw_sum[k][c];
          es += m_ring[s]->bkt_ej_sum[k][c];
        }
        double tot = n ? (double)rs / n : 0.0;
        double bw  = n ? (double)bs / n : 0.0;
        double ej  = n ? (double)es / n : 0.0;
        double wire = tot - bw - ej;   // legit hop latency = total - bw - ejwait
        printf("[RINGSRC] %-5s src_chiplet=%u ring_avg=%.1f  (bw=%.1f wire=%.1f "
               "ejwait=%.1f) [n=%llu]\n", tag, c, tot, bw, wire, ej, n);
        for (unsigned k = 0; k < ring_net::RING_CLS; ++k) {
          unsigned long long nk = m_ring[s]->ringlat_src_cnt[k][c];
          double tk = nk ? (double)m_ring[s]->ringlat_src_sum[k][c] / nk : 0.0;
          double bk = nk ? (double)m_ring[s]->bkt_bw_sum[k][c] / nk : 0.0;
          double ek = nk ? (double)m_ring[s]->bkt_ej_sum[k][c] / nk : 0.0;
          printf("[RINGSRC] %-5s class=%s src_chiplet=%u ring_avg=%.1f  (bw=%.1f "
                 "wire=%.1f ejwait=%.1f) [n=%llu]\n",
                 tag, ring_cls_name(k), c, tk, bk, tk - bk - ek, ek, nk);
        }
      }
    }
    // [XLATDBG_WIRE]: per class, the measured min hop time must equal the configured one.
    for (unsigned s = 0; s < m_ring.size(); ++s) {
      if (!m_ring[s]) continue;
      const char* tag = (s == REQ_NET) ? "Req" : "Reply";
      for (unsigned k = 0; k < ring_net::RING_CLS; ++k) {
        unsigned long long n = m_ring[s]->xlatdbg_wire_cnt[k];
        printf("[XLATDBG_WIRE] %-5s class=%s hops=%llu expected=%u min=%llu "
               "avg=%.2f max=%llu %s\n",
               tag, ring_cls_name(k), n, m_ring[s]->hop_lat_cfg(k),
               n ? m_ring[s]->xlatdbg_wire_min[k] : 0ULL,
               n ? (double)m_ring[s]->xlatdbg_wire_sum[k] / n : 0.0,
               m_ring[s]->xlatdbg_wire_max[k],
               (n == 0) ? "NO_TRAFFIC"
                        : ((m_ring[s]->xlatdbg_wire_min[k] ==
                            m_ring[s]->hop_lat_cfg(k)) ? "OK" : "MISMATCH"));
      }
    }
    printf("[RINGINJ] handoff_advances=%llu\n", m_rinj_advances);
    for (unsigned s = 0; s < m_ring.size(); ++s) {
      const char* tag = (s == REQ_NET) ? "Req" : "Reply";
      for (unsigned ch = 0; ch < m_n_chiplets; ++ch) {
        unsigned long long busy = m_rinj_egress_busy[s][ch];
        unsigned long long stall = m_rinj_stall[s][ch];
        unsigned long long ok = m_rinj_ok[s][ch];
        double adv = m_rinj_advances ? (double)m_rinj_advances : 1.0;
        printf("[RINGINJ] %-5s src_chiplet=%u egress_busy=%llu(%.1f%%) "
               "ring_full_stall=%llu(%.1f%%) injected=%llu stall/busy=%.1f%%\n",
               tag, ch, busy, 100.0 * busy / adv,
               stall, 100.0 * stall / adv, ok,
               busy ? 100.0 * stall / busy : 0.0);
      }
    }
    // [RINGLINK]: load of the link out of node c: cw goes to c+1, ccw to c-1.
    for (unsigned s = 0; s < m_ring.size(); ++s) {
      const char* tag = (s == REQ_NET) ? "Req" : "Reply";
      unsigned long long smp = m_ring[s]->link_occ_samples ? m_ring[s]->link_occ_samples : 1;
      unsigned N = m_ring[s]->ring_N();
      for (unsigned c = 0; c < N; ++c) {
        unsigned cw_dst = (c + 1) % N, ccw_dst = (c + N - 1) % N;
        printf("[RINGLINK] %-5s node%u  cw(->%u): flits=%llu occ_avg=%.2f | "
               "ccw(->%u): flits=%llu occ_avg=%.2f\n",
               tag, c, cw_dst, m_ring[s]->link_flits[c][0],
               (double)m_ring[s]->link_occ_sum[c][0] / smp,
               ccw_dst, m_ring[s]->link_flits[c][1],
               (double)m_ring[s]->link_occ_sum[c][1] / smp);
      }
    }
  }
}


void LocalInterconnect::DisplayOverallStats() const {}

unsigned LocalInterconnect::GetFlitSize() const { return LOCAL_INCT_FLIT_SIZE; }

void LocalInterconnect::DisplayState(FILE* fp) const {
  fprintf(fp, "GPGPU-Sim uArch: ICNT:Display State: Under implementation\n");
}

void LocalInterconnect::configure_chiplets(
    unsigned n_chiplets,
    const std::vector<unsigned>& cluster_to_chiplet,
    const std::vector<unsigned>& mem_sub_to_chiplet) {
  // Runs before CreateInterconnect(), so the map sizes give the device counts.
  const unsigned ns = (unsigned)cluster_to_chiplet.size();  // == n_simt_clusters
  const unsigned nm = (unsigned)mem_sub_to_chiplet.size();   // == n_mem_sub_partition
  m_n_chiplets = n_chiplets;

  const unsigned ndev = ns + nm;
  m_dev_chiplet.assign(ndev, 0);
  m_local_idx.assign(ndev, 0);
  m_n_local_cluster.assign(n_chiplets, 0);
  m_n_local_mem.assign(n_chiplets, 0);

  // 1) clusters: local idx 0,1,2,... per chiplet, in global order.
  for (unsigned d = 0; d < ns; ++d) {
    unsigned ch = cluster_to_chiplet[d];
    m_dev_chiplet[d] = ch;
    m_local_idx[d]   = m_n_local_cluster[ch]++;
  }
  // 2) mems: local idx = (that chiplet's cluster count) + 0,1,2,... (2-pass).
  for (unsigned s = 0; s < nm; ++s) {
    unsigned dev = ns + s;
    unsigned ch  = mem_sub_to_chiplet[s];
    m_dev_chiplet[dev] = ch;
    m_local_idx[dev]   = m_n_local_cluster[ch] + (m_n_local_mem[ch]++);
  }
  m_chiplets_configured = true;
}
