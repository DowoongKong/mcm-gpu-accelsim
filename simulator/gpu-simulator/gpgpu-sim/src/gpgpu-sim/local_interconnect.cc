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

// Name of a ring packet class for the RING* diagnostic rows. Index matches
// RingPkt::cls (0 = ordinary data traffic, 1 = page-table walk read/reply).
static const char* ring_cls_name(unsigned cls) { return cls ? "ptw" : "data"; }

// FUNNEL_ (temp): current gpu cycle, set by gpgpu_sim::cycle() before icnt_transfer().
// Used to stamp a TLB reply's arrival at its destination cluster out_buffer so the
// reply-icnt latency (B) can be split into transit vs last-hop pop-wait.
unsigned long long g_icnt_now = 0;

// §25 pre-measurement: split ICNT's wall-clock into "local xbar Advance()"
// (candidate for chiplet-width parallel) vs "handoff/handoff_ring" (must stay
// sequential). Defined in gpu-sim.cc (PHASE_TIMING, §24) as free-standing
// globals (not struct members) precisely so they can be extern'd here without
// needing phase_timers_t's full definition in this translation unit.
extern const bool g_phase_timing_on;
extern unsigned long long g_icnt_local_ns, g_icnt_handoff_ns;
extern unsigned long long g_icnt_local_calls, g_icnt_handoff_calls;
static inline unsigned long long icnt_split_now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

// FUNNEL_ (temp): normal (non-PTW) reply xbar latency by SOURCE mem chiplet, to
// compare against PTW reply transit (same measurement points). Indexed [src chiplet].
unsigned long long g_normrep_transit_sum[8] = {0};
unsigned long long g_normrep_lasthop_sum[8] = {0};
unsigned long long g_normrep_cnt[8]         = {0};
// FUNNEL_ (temp): normal-reply transit as a [source mem chiplet][dest cluster chiplet]
// matrix, to localize which src->dst pairs are slow (ring path vs dest ingress).
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
  m_out_mirror.assign(total_nodes, nullptr);   // H3-3b
  m_icp_voq.resize(total_nodes);          // Fix B: per-output ICP ingress VOQ
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

  // H1-A: destination-candidate outputs (see the member comment for the proof that
  // the skipped outputs cannot change any state or counter). A single-subnet build
  // carries both directions on one router, so nothing may be skipped there.
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
  // H1-B: pending-input counts, one slot per possible destination index.
  m_out_pending.assign(total_nodes, 0u);
  // H1-D: conflict-count marks, one slot per possible destination index.
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
  unsigned flits = 1;  // SMELL-13 fix: fallback matches flits_of(); the two
                       // branches below cover every current mf_type.
  // P3: when inter traffic is carried by the ring, the local xbar uses LOCAL flit
  // size for everything (no remote inflation). Ring owns inter bandwidth.
  bool is_local = m_inter_via_ring ? true : mf->is_local_access();
  if (mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_REQUEST
      || mf->get_type() == TLB_FILL_ACC)
    flits = mf->get_num_flits(true, is_local);
  else if (mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK
      || mf->get_type() == TLB_FILL_REPLY)
    flits = mf->get_num_flits(false, is_local);

  // H1-A: the candidate set is the premise of the grant-loop skip. Checked only in
  // the debug build -- is_out_candidate() is a linear scan and Push() is hot.
#ifdef XBAR_CAND_CHECK
  assert(is_out_candidate(output_deviceID) &&
         "H1-A: packet addressed to a non-candidate output");
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
  // 44-1: running total maintained by flit_queue; was a full queue copy + walk.
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
    // TSan found this one first, and it is the only xbar counter reachable from
    // the parallel CORE phase: icnt_has_buffer() -> Has_Buffer_In(.., true) is
    // called by every SM's L1 cache in pass C (gpu-cache.cc:1218). The other
    // xbar counters live in Advance(), which runs in the sequential ICNT phase.
    // The return value itself is computed from in_buffers[input_deviceID] alone,
    // so the MODEL is unaffected -- only this statistic was racing.
#pragma omp atomic
    in_buffer_full++;
  }

  return has_buffer;
}

bool xbar_router::Has_Buffer_Out(unsigned output_deviceID, unsigned flits) {
  assert(output_deviceID < total_nodes);

  // Occupancy in flits. BUG-9 fix: the iSLIP/VOQ completion paths push
  // packets with remaining_flits restored to orig_flits, so a queued egress
  // packet occupies its full size and out_buffer_limit exerts real
  // backpressure again (it was inert while completed packets carried 0).
  // The grant loop still probes with `flits == 1` before picking an input,
  // so the limit can be overshot by at most one packet -- same slack the
  // packet-count check had upstream.
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
  // issued[out] = grants given to output port `out` this cycle; up to m_port_width
  // (per-port BW = m_port_width * flit_size * icnt_freq).
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
          // FUNNEL_ (temp): see iSLIP_Advance — stamp any packet's out_buffer arrival.
          mem_fetch* _rmf = static_cast<mem_fetch*>(_packet.data);
          if (g_accelsim_diag && _rmf) _rmf->set_icnt_receive_time((unsigned)g_icnt_now);  // ST-10: DIAG-only readers
          out_buffers[_packet.output_deviceID].push(_packet);
          mark_out(_packet.output_deviceID);   // H3-3b
          in_buffers[node_id].pop();
          issued[_packet.output_deviceID]++;
          reqs++;
        } else
          conflict_sub++;
      } else {
        {  // TSan: shared xbar counter, bumped from every SM
        #pragma omp atomic
          out_buffer_full++;
        }

        if (issued[_packet.output_deviceID]) conflict_sub++;
      }
    }
  }
  next_node_id = next_node_id + 1;
  next_node_id = (next_node_id % total_nodes);

  {  // TSan: shared xbar counter, bumped from every SM
  #pragma omp atomic
    conflicts += conflict_sub;
  }
  if (active) {
    {  // TSan: shared xbar counter, bumped from every SM
    #pragma omp atomic
      conflicts_util += conflict_sub;
    }
    {  // TSan: shared xbar counter, bumped from every SM
    #pragma omp atomic
      cycles_util++;
    }
    {  // TSan: shared xbar counter, bumped from every SM
    #pragma omp atomic
      reqs_util += reqs;
    }
  }

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  // collect some stats about buffer util
  // Every node is an INPUT on this router (clusters inject in REQ_NET, mems in
  // REPLY_NET), so this loop must stay over all of them -- only the atomics go
  // (H1-C), folded into one add each since integer addition is associative.
  unsigned long long _in_u = 0, _out_u = 0;
  for (unsigned i = 0; i < total_nodes; ++i) {
    _in_u  += in_buffers[i].size();
    _out_u += out_buffers[i].size();
  }
  in_buffer_util  += _in_u;
  out_buffer_util += _out_u;

  cycles++;   // H1-C: router-private counter
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
  // H1-D: the old form kept the destinations seen so far in a heap-allocated vector
  // and ran std::find over it per input (O(N^2) worst case, one malloc per cycle per
  // router). A per-cycle stamp gives the same verdict -- "first input to name this
  // destination" vs "a later one" -- in O(1) per input with no allocation.
  ++m_seen_cur;
  // H1-B: rebuild the pending-input counts in the same pass (no extra walk).
  std::fill(m_out_pending.begin(), m_out_pending.end(), 0u);
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) {
      const unsigned _tgt = in_buffers[i].front().output_deviceID;
      m_out_pending[_tgt]++;                       // H1-B
      if (m_seen_stamp[_tgt] == m_seen_cur)
        conflict_sub++;
      else
        m_seen_stamp[_tgt] = m_seen_cur;
      active = true;
    }
  }

  // H1-C: these counters are written ONLY inside Advance()/iSLIP_Advance()/
  // RR_Advance(), and LocalInterconnect::Advance() gives one router to exactly one
  // thread (the parallel loop is over net[subnet][chiplet], all private state), so
  // they are not shared and need no atomic. in_buffer_full is the one exception --
  // Has_Buffer_In() is called by every SM thread -- and it keeps its atomic.
  conflicts += conflict_sub;
  if (active) {
    conflicts_util += conflict_sub;
    cycles_util++;
  }
  // 37: when no input flit exists anywhere and the ICP VOQ is empty, the
  // grant loop below cannot grant, and every piece of iSLIP state it owns
  // (next_node[i], m_icp_voq_rr[i], grant_cycles_count) moves ONLY on a grant.
  // Its single observable effect in this state is the out_buffer_full sample
  // for outputs whose out-buffer is full -- replicate that and skip the
  // O(outputs x port_width x inputs) scan. Tail stats below run unchanged.
  if (!active && m_icp_voq_flits == 0) {
    unsigned obf = 0;
    // H1-A: a non-candidate output's out_buffer is never pushed to, so it is always
    // empty and Has_Buffer_Out() is always true for it -- it could never be counted.
    for (unsigned oi = 0; oi < m_out_cand.size(); ++oi)
      if (!Has_Buffer_Out(m_out_cand[oi], 1)) obf++;
    if (obf) out_buffer_full += obf;   // H1-C: router-private counter
  } else {
  // do iSLIP. Each OUTPUT port may grant up to m_port_width flits per cycle (spanning
  // packets/inputs) -> per-port BW = m_port_width * flit_size * icnt_freq.
  for (unsigned oi = 0; oi < m_out_cand.size(); ++oi) {
   const unsigned i = m_out_cand[oi];   // H1-A: skipped outputs can never be granted
   for (unsigned w = 0; w < m_port_width; ++w) {
    bool _did_grant = false;
    if (Has_Buffer_Out(i, 1)) {
      // H1-B: no input's front is addressed to this output and its ICP VOQ is empty,
      // so both passes below would fail and fall through to the break -- and that
      // path touches nothing (next_node[i] and m_icp_voq_rr[i] move only on a grant,
      // out_buffer_full only when Has_Buffer_Out fails, which it did not). Skipping
      // is therefore state- and counter-identical to running the scan.
      if (m_out_pending[i] == 0 && m_icp_voq[i].empty()) {
#ifdef XBAR_CAND_CHECK   // H1-B: no input front may be addressed to i
        for (unsigned _j = 0; _j < total_nodes; ++_j)
          assert(in_buffers[_j].empty() || in_buffers[_j].front().output_deviceID != i);
#endif
        break;
      }
      // Fix B: RR-alternate ICP VOQ[i] vs local inputs (starvation-free). pass 0 =
      // preferred source (toggled per grant), pass 1 = fallback to the other.
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
              if (g_accelsim_diag && _rmf) _rmf->set_icnt_receive_time((unsigned)g_icnt_now);  // ST-10: DIAG-only readers
              Packet _done = _vp;
              _done.remaining_flits = _done.orig_flits;  // BUG-9 fix: egress
              out_buffers[i].push(_done);                // occupancy = pkt size
              mark_out(i);                               // H3-3b
              m_icp_voq[i].pop();
            }
            reqs++;
            _did_grant = true;
          }
        } else {
          // (B) serve a local input (unchanged iSLIP scan over in_buffers)
          for (unsigned j = 0; j < total_nodes; ++j) {
            unsigned node_id = (j + next_node[i]) % total_nodes;
            if (!in_buffers[node_id].empty()) {
              const Packet &_packet = in_buffers[node_id].front();
              if (_packet.output_deviceID == i) {
                in_buffers[node_id].dec_front_flits();   // 44-1: keeps the running total
                if (_packet.remaining_flits == 0) {
                  mem_fetch* _rmf = static_cast<mem_fetch*>(_packet.data);
                  if (g_accelsim_diag && _rmf) _rmf->set_icnt_receive_time((unsigned)g_icnt_now);  // ST-10: DIAG-only readers
                  Packet _done = _packet;
                  _done.remaining_flits = _done.orig_flits;  // BUG-9 fix:
                  out_buffers[_packet.output_deviceID].push(_done);  // full size
                  mark_out(_packet.output_deviceID);                 // H3-3b
                  in_buffers[node_id].pop();
                  // H1-B: this input's front just changed. The packet that left was
                  // addressed to `i` (that is why it was granted), so drop one from
                  // i's count and credit the new front, if any. Keeps the counts
                  // exact for the remaining outputs of this same cycle.
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
      out_buffer_full++;      // H1-C: router-private counter
    if (!_did_grant) break;   // out buffer full or no packet for this output -> stop
   }
  }

  }  // 37: end of skip-else

  if (active) reqs_util += reqs;   // H1-C: router-private counter

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

  // collect some stats about buffer util
  // Every node is an INPUT on this router (clusters inject in REQ_NET, mems in
  // REPLY_NET), so this loop must stay over all of them -- only the atomics go
  // (H1-C), folded into one add each since integer addition is associative.
  unsigned long long _in_u = 0, _out_u = 0;
  for (unsigned i = 0; i < total_nodes; ++i) {
    _in_u  += in_buffers[i].size();
    _out_u += out_buffers[i].size();
  }
  in_buffer_util  += _in_u;
  out_buffer_util += _out_u;

  cycles++;   // H1-C: router-private counter
}

bool xbar_router::Busy() const {
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) return true;

    if (!out_buffers[i].empty()) return true;

    if (!m_icp_voq[i].empty()) return true;   // Fix B: pending ICP ingress VOQ packets
  }
  return false;
}

////////////////////////////////////////////////////
/////////////LocalInterconnect/////////////////////

// assume all the packets are one flit
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
  // n_shader/n_mem are set now -> validate the injected maps here (NOT in
  // configure_chiplets, where they are still 0).
  assert(m_chiplets_configured && "configure_chiplets must run before CreateInterconnect");
  assert(m_dev_chiplet.size() == n_shader + n_mem && "chiplet map size mismatch");

  m_use_ring = (m_inct_config.use_ring != 0) && (m_n_chiplets > 1);  // MOVED UP (needed below)
  const unsigned n_icp = (m_n_chiplets > 1) ? 1u : 0u;
  net.resize(n_subnets);
  for (unsigned s = 0; s < n_subnets; ++s) {
    net[s].resize(m_n_chiplets, nullptr);
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch) {
      net[s][ch] = new xbar_router(ch, static_cast<Interconnect_type>(s),
                                   m_n_local_cluster[ch], m_n_local_mem[ch],
                                   m_inct_config, n_icp, m_use_ring,  // P3: inter_via_ring
                                   /*both_dirs=*/(n_subnets == 1));   // H1-A
    }
  }
  if (m_use_ring) {
    m_ring.assign(n_subnets, nullptr);
    // A negative -icnt_ptw_ring_hop_latency means "same as the data class", which keeps
    // the default byte-identical to the single-latency ring.
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
                               m_inct_config.ring_link_width);   // P4: BW
    // RINGINJ_ (temp): [subnet][chiplet] source-side egress accounting.
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
    assert(net[subnet][src_ch]->Has_Buffer_In(m_local_idx[in], size));
    net[subnet][src_ch]->Push(m_local_idx[in], m_local_idx[out], data, size);
  } else {                                                     // REMOTE
    assert(net[subnet][src_ch]->Has_Buffer_In(m_local_idx[in], size));
    net[subnet][src_ch]->Push(m_local_idx[in], net[subnet][src_ch]->icp_index(), data, size);
  }
}

// H3-3b: Pop() 과 똑같이 소비자 쪽 규칙으로 서브넷·칩렛·로컬 인덱스를 고른다.
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
  // Subnet chosen by the CONSUMER side, same rule as the original code.
  unsigned subnet = (out < n_shader) ? REPLY_NET : REQ_NET;
  if (n_subnets == 1) subnet = 0;
  unsigned ch = m_dev_chiplet[out];
  return net[subnet][ch]->Pop(m_local_idx[out]);
}

void LocalInterconnect::Advance() {
  // §25 pre-measurement (diagnostic only, gated off by default -- see
  // gpu-sim.cc PHASE_TIMING/§24): split into "local" (each net[s][ch] is a
  // chiplet-private xbar_router -- the §25 chiplet-width parallel candidate)
  // and "handoff" (ring advance + handoff/handoff_ring -- genuine
  // chiplet-to-chiplet exchange, must stay sequential either way).
  // P25: chiplet-width parallel, same T5 independence argument already used for
  // CORE Phase B and the L2/GMMU loop -- net[s][ch] is a chiplet-private
  // xbar_router (own in_buffers/out_buffers/counters, §25-0). Advance() runs in
  // the sequential ICNT clock domain (local_interconnect.cc:154's own comment:
  // "the other xbar counters live in Advance(), which runs in the sequential
  // ICNT phase"), so this loop never overlaps Phase A/B/C's parallel SM work --
  // only with itself, across chiplets, which is exactly what T5 covers.
  // §25-3 E5 (2026-08-15, applied after the chiplet-only version passed E1-E4):
  // collapse subnet(REQ_NET/REPLY_NET) together with chiplet. net[REQ_NET][ch]
  // and net[REPLY_NET][ch] are separate xbar_router instances with no shared
  // state between them (same disjointness argument as the chiplet axis), so
  // up to n_subnets*m_n_chiplets=8 iterations can run at once instead of being
  // capped at m_n_chiplets=4 per subnet, processed in two sequential batches.
  const unsigned long long __icnt_t0 = g_phase_timing_on ? icnt_split_now_ns() : 0;
#pragma omp parallel for schedule(runtime) collapse(2)
  for (unsigned s = 0; s < n_subnets; ++s)
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch)
      net[s][ch]->Advance();
  if (g_phase_timing_on) {
    g_icnt_local_ns += icnt_split_now_ns() - __icnt_t0;
    g_icnt_local_calls++;
  }

  const unsigned long long __icnt_t1 = g_phase_timing_on ? icnt_split_now_ns() : 0;
  if (m_n_chiplets > 1) {
    if (m_use_ring) {
      for (unsigned s = 0; s < n_subnets; ++s) m_ring[s]->advance();
      handoff_ring();
    } else {
      handoff();              // Phase 1 zero-latency direct path
    }
  }
  if (g_phase_timing_on) {
    g_icnt_handoff_ns += icnt_split_now_ns() - __icnt_t1;
    g_icnt_handoff_calls++;
  }
}

void LocalInterconnect::handoff() {
  for (unsigned s = 0; s < n_subnets; ++s) {
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch) {
      xbar_router* src = net[s][ch];
      unsigned egress = src->icp_index();
      // Drain the ICP egress toward its destination chiplet (multiple packets
      // per Advance allowed; capping at 1 would also be a valid model).
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
        (void) src->Pop(egress);                        // 실제 제거
        dst->Push(ingress, m_local_idx[final_dev], data, 0);  // size ignored (flits from mf)
      }
    }
  }
}

void LocalInterconnect::handoff_ring() {
  if (g_accelsim_diag) m_rinj_advances++;   // RINGINJ_ (temp, P-5 gated)
  for (unsigned s = 0; s < n_subnets; ++s) {
    ring_net* ring = m_ring[s];

    // (1) inject: each chiplet's ICP egress -> ring
    for (unsigned ch = 0; ch < m_n_chiplets; ++ch) {
      xbar_router* src = net[s][ch];
      unsigned egress = src->icp_index();
      bool had_pending = src->HasOut(egress);   // RINGINJ_ (temp)
      bool stalled = false;                     // RINGINJ_ (temp)
      while (src->HasOut(egress)) {
        void* data = src->TopOut(egress);
        mem_fetch* mf = static_cast<mem_fetch*>(data);
        unsigned final_dev = (s == REQ_NET)
            ? (n_shader + mf->get_sub_partition_id())   // mem2device
            : (mf->get_tpc());                          // cluster deviceID
        unsigned dch   = m_dev_chiplet[final_dev];
        // Ring link width = inter flit size (NOT the local xbar's local size).
        unsigned flits = mf->get_num_flits(s == REQ_NET, /*is_local=*/false);
        // Page-table-walk packets are tagged by the GMMU (gmmu.cc sets TLB_FILL_ACC) and
        // keep that type on the reply leg, so both directions land in class 1.
        const bool is_ptw = (mf->get_type() == TLB_FILL_ACC ||
                             mf->get_type() == TLB_FILL_REPLY);
        if (!ring->inject(ch, dch, final_dev, data, flits, is_ptw ? 1u : 0u)) {
          stalled = true; break;   // link full
        }
        (void) src->Pop(egress);
        if (g_accelsim_diag) m_rinj_ok[s][ch]++;  // RINGINJ_ (temp, P-5 gated)
      }
      // RINGINJ_ (temp): egress had work this cycle; stalled = blocked by ring-link-full.
      if (g_accelsim_diag && had_pending) {  // P-5 gated
        m_rinj_egress_busy[s][ch]++;
        if (stalled) m_rinj_stall[s][ch]++;
      }
    }

    // (2) eject: ring -> dst chiplet's ICP ingress VOQ (Fix B: per-destination queue,
    // removes head-of-line blocking; packets still traverse iSLIP per-flit downstream).
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

bool LocalInterconnect::HasBuffer(unsigned deviceID, unsigned int size) const {
  unsigned subnet = (n_subnets > 1 && deviceID >= n_shader) ? REPLY_NET : REQ_NET;
  unsigned ch     = m_dev_chiplet[deviceID];
  return net[subnet][ch]->Has_Buffer_In(m_local_idx[deviceID], size, true);
}

void LocalInterconnect::DisplayStats() const {
  // Every router Advance()s each cycle, so 'cycles' is identical across
  // chiplets -> use chiplet 0 as representative for *_per_cycle ratios.
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
  // RING* (temp) report -- P-5: only under ACCELSIM_DIAG (counters gated too).
  if (m_use_ring && g_accelsim_diag) {
    for (unsigned s = 0; s < m_ring.size(); ++s) {
      if (!m_ring[s]) continue;
      const char* tag = (s == REQ_NET) ? "Req" : "Reply";
      // Combined row first (unchanged format, so older logs stay comparable), then
      // one row per packet class. class=data must not move when only the page-table
      // walk latency is changed; that row is the evidence for "data traffic untouched".
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
    // RINGINJ_ (temp): source-side ring egress accounting per (subnet, src chiplet).
    // egress_busy%% high + stall%% high  -> ring-link BW saturated at that source.
    // egress_busy%% high + stall%% low   -> xbar arbitration can't fill egress fast enough.
    // RINGSRC_ (temp): per-source-chiplet ring residency (inject->eject). If a source's
    // avg is high while links are near-empty, the wait is at the hub eject (order unfairness).
    for (unsigned s = 0; s < m_ring.size(); ++s) {
      const char* tag = (s == REQ_NET) ? "Req" : "Reply";
      unsigned N = m_ring[s]->ring_N();
      for (unsigned c = 0; c < N; ++c) {
        // Combined row (unchanged format), then one row per packet class. `wire` is
        // derived by subtraction, so mixing classes with different hop latencies
        // makes the combined `wire` a blend that answers no question on its own.
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
    // XLATDBG_ (temp): self-check for the per-class hop latency. `expected` is the
    // configured value (an independent source: the config literal), `min` is measured
    // directly from wire entry to delivery. They must be equal; anything else means
    // the wire stage is not applying the configured latency to that class.
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
    // RINGLINK_ (temp): per-(node,dir) link load. Reveals tie-break CW/CCW imbalance:
    // the CW link out of the hub carries both 1-hop and 2-hop(equidistant, tie->cw) traffic.
    // link (node c, dir d) = the wire from c to neighbor(c,d): d0=cw (c->c+1), d1=ccw (c->c-1).
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
  // Runs BEFORE CreateInterconnect, so n_shader/n_mem are not set yet.
  // The map sizes are the authoritative device counts.
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
