#pragma once
#ifdef __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

using u64 = uint64_t;
using u32 = uint32_t;
using int64 = int64_t;

namespace epic
{
  namespace gpu
  {

    __global__ void create_positions(
        u64 *start_positions,
        u64 *positions_and_results,
        u64 number_of_start_positions,
        u32 k)
    {
      u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < number_of_start_positions)
      {
        u64 first_position = start_positions[idx];
        u64 last_position = start_positions[idx + 1ULL] - k;
        u32 row_length = last_position - first_position + 1ULL;
        u64 offset = first_position - (idx * (k - 1));
        for (u32 i = 0U; i < row_length; i++)
        {
          positions_and_results[offset + i] = first_position + i;
        }
      }
    }

  }
}