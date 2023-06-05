#pragma once
#ifdef __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include "../enums.hpp"
#include "rank_kernel.hpp"
using u64 = uint64_t;
using u32 = uint32_t;

namespace epic
{
  namespace gpu
  {

    // Pre-search all k-mers of size m.
    template <int superblock_size, bool shuffles = false, int rank_version = epic::kind::poppy>
    __global__ void presearch(
        u64 *C, u64 **GBWT, u64 **L0, u64 **L12, u64 *out_left, u64 *out_right, u32 m)
    {
      u32 idx = (u32)(blockIdx.x * blockDim.x + threadIdx.x);
      u32 kmer = idx;
      u32 c = (kmer >> ((m - 1U) * 2U)) & 0b11U;
      u64 node_left = C[c];
      u64 node_right = C[c + 1] - 1ULL;

      for (int i = ((m - 2U) * 2U); i >= 0; i -= 2)
      {
        c = (kmer >> i) & 0b11U;
        node_left = C[c] + epic::gpu::rank<superblock_size, shuffles, rank_version>(GBWT[c], L0[c], L12[c], node_left);
        node_right = C[c] + epic::gpu::rank<superblock_size, shuffles, rank_version>(GBWT[c], L0[c], L12[c], node_right + 1ULL) - 1ULL;
      }
      out_left[idx] = node_left;
      out_right[idx] = node_right;
    }

  }
}
