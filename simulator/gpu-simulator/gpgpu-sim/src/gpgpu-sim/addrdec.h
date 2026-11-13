// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../option_parser.h"

#ifndef ADDRDEC_H
#define ADDRDEC_H

#include "../abstract_hardware_model.h"

enum partition_index_function {
  CONSECUTIVE = 0,
  BITWISE_PERMUTATION,
  IPOLY,
  PAE,
  RANDOM,
  CUSTOM,
  // 6 is IPOLY_MODULO in Accel-Sim 2.0. It is left unused here so that a config
  // written for that simulator cannot silently select a different function.
  CHIPLET_IPOLY = 7  // IPOLY over the sub-partitions of one chiplet; chiplet kept
};

struct addrdec_t {
  void print(FILE *fp) const;

  unsigned chip;
  unsigned bk;
  unsigned row;
  unsigned col;
  unsigned burst;

  unsigned sub_partition;
};

class linear_to_raw_address_translation {
 public:
  linear_to_raw_address_translation();
  void addrdec_setoption(option_parser_t opp);
  void init(unsigned int n_channel, unsigned int n_sub_partition_in_channel);

  // accessors
  void addrdec_tlx(new_addr_type addr, addrdec_t *tlx) const;
  new_addr_type partition_address(new_addr_type addr) const;
  // S2 (llm_exp/docs/s31_sim_speed_review_llm.ko.md): true when addrdec_tlx()'s
  // sub_partition cannot change inside an aligned 2^line_bits-byte block, i.e. no
  // chip or bank bit and no hash input lies below line_bits, and no RNG is used.
  bool sub_partition_fixed_within(unsigned line_bits) const {
    if (memory_partition_indexing == RANDOM) return false;
    if (ADDR_CHIP_S >= 0 && (unsigned)ADDR_CHIP_S < line_bits) return false;
    const new_addr_type low = ((new_addr_type)1 << line_bits) - 1;
    return ((addrdec_mask[CHIP] | addrdec_mask[BK]) & low) == 0;
  }
  // CHIPLET_IPOLY: gpu_module checks hash_chiplets() against -gpgpu_n_chiplets.
  bool chiplet_hash_on() const { return memory_partition_indexing == CHIPLET_IPOLY; }
  unsigned hash_chiplets() const { return m_hash_chiplets; }

 private:
  void addrdec_parseoption(const char *option);
  void sweep_test() const;  // sanity check to ensure no overlapping

  enum { CHIP = 0, BK = 1, ROW = 2, COL = 3, BURST = 4, N_ADDRDEC };

  const char *addrdec_option;
  int gpgpu_mem_address_mask;
  partition_index_function memory_partition_indexing;
  bool run_test;

  int ADDR_CHIP_S;
  unsigned char addrdec_mklow[N_ADDRDEC];
  unsigned char addrdec_mkhigh[N_ADDRDEC];
  new_addr_type addrdec_mask[N_ADDRDEC];
  new_addr_type sub_partition_id_mask;

  unsigned int gap;
  unsigned m_n_channel;
  int m_n_sub_partition_in_channel;
  int m_n_sub_partition_total;
  unsigned log2channel;
  unsigned log2sub_partition;
  unsigned nextPowerOf2_m_n_channel;

  // CHIPLET_IPOLY: number of chiplets whose sub-partition ranges the hash keeps
  // apart (-gpgpu_mem_hash_chiplets), and sub-partitions per chiplet (set in init).
  unsigned m_hash_chiplets;
  unsigned m_slices_per_chiplet;
};

#endif
