#pragma once
#ifdef __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include "../enums.hpp"
#include "rank_kernel.hpp"

typedef uint64_t u64;
typedef uint32_t u32;

namespace epic
{
  namespace gpu
  {
    template <int superblock_size, bool shuffles = false, int rank_version = epic::kind::poppy, bool with_presearch = true>
    __global__ void search_unified(
        u64 *kmerstring, u64 *C, u64 **SBWT, u64 **L0, u64 **L12, u64 *node_lefts, u64 *node_rights, u32 kmer_size, u64 *positions_in_and_results_out, u32 m)
    {

      u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
      u64 kmerIndex = positions_in_and_results_out[idx] << 1;
      u64 kmer =
          (kmerstring[kmerIndex >> 6] << (kmerIndex & 63U)) | ((kmerstring[(kmerIndex >> 6) + 1ULL] >> (63U - (kmerIndex & 63U))) >> 1);
      u32 c;
      u64 node_left, node_right;
      if (with_presearch)
      {
        u32 m_mer = kmer >> (64 - (m * 2U));
        node_left = node_lefts[m_mer];
        node_right = node_rights[m_mer];
      }
      else
      {
        node_left = 0ULL;
        node_right = C[4] - 1ULL; // The same as node_right = n - 1, if n is the number of k-mers in the SBWT index.
        m = 0U;
      }

      for (u32 i = m; i < kmer_size; i += 1U)
      {
        c = (kmer >> (62U - (i << 1))) & 0b11U;
        node_left = C[c] + epic::gpu::rank<superblock_size, shuffles, rank_version>(SBWT[c], L0[c], L12[c], node_left);
        node_right = C[c] + epic::gpu::rank<superblock_size, shuffles, rank_version>(SBWT[c], L0[c], L12[c], node_right + 1ULL) - 1ULL;
        if (!shuffles)
        {
          if (node_left > node_right)
          {
            break;
          }
	}
      }
      if (node_left > node_right)
        node_left = ~0ULL;

      positions_in_and_results_out[idx] = node_left;
    }

    template <int superblock_size, bool shuffles = false, int rank_version = epic::kind::poppy, bool with_presearch = true>
    __global__ void search(
        u64 *kmerstring, u64 *C, u64 **SBWT, u64 **L0, u64 **L12, u64 *node_lefts, u64 *node_rights, u32 kmer_size, u64 *positions_in_and_results_out, u32 m)
    {
      u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
      u64 kmerIndex = positions_in_and_results_out[idx] << 1;
      u64 kmer =
          (kmerstring[kmerIndex >> 6] << (kmerIndex & 63U)) | ((kmerstring[(kmerIndex >> 6) + 1ULL] >> (63U - (kmerIndex & 63U))) >> 1);
      u32 c;
      u64 node_left, node_right;
      if (with_presearch)
      {
        u32 m_mer = kmer >> (64 - (m * 2U));
        node_left = node_lefts[m_mer];
        node_right = node_rights[m_mer];
      }
      else
      {
        node_left = 0ULL;
        node_right = C[4] - 1ULL;
        m = 0U;
      }

      for (u32 i = m; i < kmer_size; i += 1U)
      {
        c = (kmer >> (62U - (i << 1))) & 0b11U;
        node_left = C[c] + epic::gpu::rank<superblock_size, shuffles, rank_version>(SBWT[c], L0[c], L12[c], node_left);
        node_right = C[c] + epic::gpu::rank<superblock_size, shuffles, rank_version>(SBWT[c], L0[c], L12[c], node_right + 1ULL) - 1ULL;
        if (!shuffles)
        {
          if (node_left > node_right)
          {
	    break;
          }
        }
      }

      if (node_left > node_right)
      {
        node_left = ~0ULL;
      }

      positions_in_and_results_out[idx] = node_left;
    }

  }
}
