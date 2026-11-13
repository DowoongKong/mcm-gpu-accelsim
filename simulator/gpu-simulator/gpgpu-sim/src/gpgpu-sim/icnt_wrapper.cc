// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
// The University of British Columbia
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

#include "icnt_wrapper.h"
#include <assert.h>
#include "../intersim2/globals.hpp"
#include "../intersim2/interconnect_interface.hpp"
#include "local_interconnect.h"
#include "par_serial_probe.h"

icnt_create_p icnt_create;
icnt_init_p icnt_init;
icnt_has_buffer_p icnt_has_buffer;
icnt_push_p icnt_push;
icnt_pop_p icnt_pop;
icnt_transfer_p icnt_transfer;
icnt_busy_p icnt_busy;
icnt_display_stats_p icnt_display_stats;
icnt_display_overall_stats_p icnt_display_overall_stats;
icnt_display_state_p icnt_display_state;
icnt_get_flit_size_p icnt_get_flit_size;

unsigned g_network_mode;
char* g_network_config_filename;

struct inct_config g_inct_config;
LocalInterconnect* g_localicnt_interface;

#include "../option_parser.h"

// Wrapper to intersim2 to accompany old icnt_wrapper
// TODO: use delegate/boost/c++11<funtion> instead

static void intersim2_create(unsigned int n_shader, unsigned int n_mem) {
  g_icnt_interface->CreateInterconnect(n_shader, n_mem);
}

static void intersim2_init() { g_icnt_interface->Init(); }

static bool intersim2_has_buffer(unsigned input, unsigned int size) {
  return g_icnt_interface->HasBuffer(input, size);
}

static void intersim2_push(unsigned input, unsigned output, void* data,
                           unsigned int size) {
  g_icnt_interface->Push(input, output, data, size);
}

static void* intersim2_pop(unsigned output) {
  return g_icnt_interface->Pop(output);
}

static void intersim2_transfer() { g_icnt_interface->Advance(); }

static bool intersim2_busy() { return g_icnt_interface->Busy(); }

static void intersim2_display_stats() { g_icnt_interface->DisplayStats(); }

static void intersim2_display_overall_stats() {
  g_icnt_interface->DisplayOverallStats();
}

static void intersim2_display_state(FILE* fp) {
  g_icnt_interface->DisplayState(fp);
}

static unsigned intersim2_get_flit_size() {
  return g_icnt_interface->GetFlitSize();
}

//////////////////////////////////////////////////////

static void LocalInterconnect_create(unsigned int n_shader,
                                     unsigned int n_mem) {
  g_localicnt_interface->CreateInterconnect(n_shader, n_mem);
}

static void LocalInterconnect_init() { g_localicnt_interface->Init(); }

// The par_probe_guard on these three is the ACCELSIM_SERIAL_SUBSYS=icnt probe:
// 3 of the 4 most frequent same-phase TSan sites sit on this path, reached from
// every SM's L1I during phase C. Inert unless the probe is selected.
static bool LocalInterconnect_has_buffer(unsigned input, unsigned int size) {
  par_probe_guard g(PAR_PROBE_ICNT);
  return g_localicnt_interface->HasBuffer(input, size);
}

static void LocalInterconnect_push(unsigned input, unsigned output, void* data,
                                   unsigned int size) {
  par_probe_guard g(PAR_PROBE_ICNT);
  g_localicnt_interface->Push(input, output, data, size);
}

static void* LocalInterconnect_pop(unsigned output) {
  par_probe_guard g(PAR_PROBE_ICNT);
  return g_localicnt_interface->Pop(output);
}

static void LocalInterconnect_transfer() { g_localicnt_interface->Advance(); }

static bool LocalInterconnect_busy() { return g_localicnt_interface->Busy(); }

static void LocalInterconnect_display_stats() {
  g_localicnt_interface->DisplayStats();
}

static void LocalInterconnect_display_overall_stats() {
  g_localicnt_interface->DisplayOverallStats();
}

static void LocalInterconnect_display_state(FILE* fp) {
  g_localicnt_interface->DisplayState(fp);
}

static unsigned LocalInterconnect_get_flit_size() {
  return g_localicnt_interface->GetFlitSize();
}

///////////////////////////

void icnt_reg_options(class OptionParser* opp) {
  option_parser_register(opp, "-network_mode", OPT_INT32, &g_network_mode,
                         "Interconnection network mode", "1");
  option_parser_register(opp, "-inter_config_file", OPT_CSTR,
                         &g_network_config_filename,
                         "Interconnection network config file", "mesh");

  // parameters for local xbar
  option_parser_register(opp, "-icnt_in_buffer_limit", OPT_UINT32,
                         &g_inct_config.in_buffer_limit, "in_buffer_limit",
                         "64");
  option_parser_register(opp, "-icnt_out_buffer_limit", OPT_UINT32,
                         &g_inct_config.out_buffer_limit, "out_buffer_limit",
                         "64");
  option_parser_register(opp, "-icnt_subnets", OPT_UINT32,
                         &g_inct_config.subnets, "subnets", "2");
  option_parser_register(opp, "-icnt_arbiter_algo", OPT_UINT32,
                         &g_inct_config.arbiter_algo, "arbiter_algo", "1");
  option_parser_register(opp, "-icnt_verbose", OPT_UINT32,
                         &g_inct_config.verbose, "inct_verbose", "0");
  option_parser_register(opp, "-icnt_grant_cycles", OPT_UINT32,
                         &g_inct_config.grant_cycles, "grant_cycles", "1");
  
  option_parser_register(opp, "-icnt_use_ring", OPT_UINT32,
                         &g_inct_config.use_ring,
                         "1 = inter-chiplet ring, 0 = direct handoff", "0");
  option_parser_register(opp, "-icnt_ring_buf", OPT_UINT32,
                         &g_inct_config.ring_buf,
                         "ring per-link/eject buffer depth (packets)", "64");
  option_parser_register(opp, "-icnt_ring_hop_latency", OPT_UINT32,
                         &g_inct_config.ring_hop_latency,
                         "ring per-hop latency in cycles (paper: 32)", "32");
  option_parser_register(opp, "-icnt_ptw_ring_hop_latency", OPT_INT32,
                         &g_inct_config.ring_hop_latency_ptw,
                         "ring per-hop latency in cycles for page-table-walk packets "
                         "(TLB_FILL_ACC/REPLY); -1 = same as -icnt_ring_hop_latency",
                         "-1");
  option_parser_register(opp, "-icnt_ring_link_width", OPT_UINT32,
                         &g_inct_config.ring_link_width,
                         "ring flits drained per cycle per link (BW = width*flit*freq)", "1");
  // Default 5: per-port BW = 5 * 40B * 1.395GHz = 279 GB/s >= HBM2 stack (256 GB/s),
  // so a single intra-chiplet xbar port can sink a chiplet's HBM2 bandwidth.
  option_parser_register(opp, "-icnt_xbar_port_width", OPT_UINT32,
                         &g_inct_config.xbar_port_width,
                         "intra-chiplet xbar flits drained per cycle per OUTPUT port "
                         "(per-port BW = width*flit*freq)", "5");
}

void icnt_wrapper_init() {
  switch (g_network_mode) {
    case INTERSIM:
      // FIXME: delete the object: may add icnt_done wrapper
      g_icnt_interface = InterconnectInterface::New(g_network_config_filename);
      icnt_create = intersim2_create;
      icnt_init = intersim2_init;
      icnt_has_buffer = intersim2_has_buffer;
      icnt_push = intersim2_push;
      icnt_pop = intersim2_pop;
      icnt_transfer = intersim2_transfer;
      icnt_busy = intersim2_busy;
      icnt_display_stats = intersim2_display_stats;
      icnt_display_overall_stats = intersim2_display_overall_stats;
      icnt_display_state = intersim2_display_state;
      icnt_get_flit_size = intersim2_get_flit_size;
      break;
    case LOCAL_XBAR:
      g_localicnt_interface = LocalInterconnect::New(g_inct_config);
      icnt_create = LocalInterconnect_create;
      icnt_init = LocalInterconnect_init;
      icnt_has_buffer = LocalInterconnect_has_buffer;
      icnt_push = LocalInterconnect_push;
      icnt_pop = LocalInterconnect_pop;
      icnt_transfer = LocalInterconnect_transfer;
      icnt_busy = LocalInterconnect_busy;
      icnt_display_stats = LocalInterconnect_display_stats;
      icnt_display_overall_stats = LocalInterconnect_display_overall_stats;
      icnt_display_state = LocalInterconnect_display_state;
      icnt_get_flit_size = LocalInterconnect_get_flit_size;
      break;
    default:
      assert(0);
      break;
  }
}

void icnt_configure_chiplets(unsigned n_chiplets,
                             const std::vector<unsigned>& cluster_to_chiplet,
                             const std::vector<unsigned>& mem_sub_to_chiplet) {
  // Only the LocalInterconnect backend is split per chiplet (Phase 1).
  if (g_network_mode == LOCAL_XBAR && g_localicnt_interface)
    g_localicnt_interface->configure_chiplets(n_chiplets, cluster_to_chiplet,
                                              mem_sub_to_chiplet);
}

bool icnt_ring_enabled() {
  return g_network_mode == LOCAL_XBAR && g_inct_config.use_ring != 0;
}

// H3-3b (speed): see icnt_wrapper.h
bool icnt_out_mirror_supported() {
  return g_network_mode == LOCAL_XBAR && g_localicnt_interface != NULL;
}
void icnt_set_out_mirror(unsigned out, unsigned char *slot) {
  g_localicnt_interface->SetOutMirror(out, slot);
}
bool icnt_out_empty(unsigned out) { return g_localicnt_interface->OutEmpty(out); }
