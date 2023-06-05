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

    __device__ u8 bytes[4][4] = {{0, 0, 0, 0}, {64, 16, 4, 1}, {128, 32, 8, 2}, {192, 48, 12, 3}};

    __device__ u8 bits[256] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    __global__ void create_kmer_string(
        u8 *raw_string,
        u64 *compact_string,
        u64 number_of_words)
    {
      u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < number_of_words)
      {
        u8 word[8] = {};
#pragma unroll
        for (u32 j = 0; j < 8U; j += 1U)
        {
          uchar4 four_letters = reinterpret_cast<uchar4 *>(raw_string)[(idx * 8) + j];
          word[7U - j] = bytes[bits[four_letters.x]][0] |
                         bytes[bits[four_letters.y]][1] |
                         bytes[bits[four_letters.z]][2] |
                         bytes[bits[four_letters.w]][3];
        }
        compact_string[idx] = reinterpret_cast<u64 *>(word)[0];
      }
    }
  }
}