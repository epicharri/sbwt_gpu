#pragma once
#include "../include/globals.hpp"
#include "../include/gpu/cuda.hpp"
#include "../include/gpu/device_memory.hpp"
// #include "../include/gpu/kernel_launchers.hpp"
// #include "../include/gpu/presearch_kernel.hpp"
#include "../include/gpu/search_and_presearch_launchers.hpp"
#include "../include/parameters.hpp"

namespace epic
{
  struct PreSearch
  {
    u32 kmer_size_presearch = 1ULL;
    u32 number_of_k_mers = 4ULL;
    u32 number_of_words = 0ULL;
    float millis_pre = 0.0;
    bool success = false;
    PreSearch(u32 t_kmer_size_presearch);
    int call_presearch(
        epic::Parameters &parameters, epic::gpu::DeviceMemory &device_memory, bool device_is_nvidia_a100);
  };

  PreSearch::PreSearch(u32 t_kmer_size_presearch)
  {
    kmer_size_presearch = t_kmer_size_presearch;
    number_of_k_mers = 1U << (2 * t_kmer_size_presearch);
    number_of_words = number_of_k_mers;
    millis_pre = 0.0;
  }

  int PreSearch::call_presearch(epic::Parameters &parameters, epic::gpu::DeviceMemory &dm, bool device_is_nvidia_a100)
  {
    if (device_is_nvidia_a100)
    {
      cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, parameters.device_set_limit_presearch); // only in Nvidia A100
    }
    u64 block_size, grid_size;
    block_size = parameters.threads_per_block;
    grid_size = number_of_k_mers / block_size;
    if (grid_size == 0ULL)
      grid_size = 1ULL;
    int rank_version = parameters.rank_structure_version;
    if (parameters.with_presearch)
    {
      switch (parameters.bits_in_superblock)
      {
      case 256:
        if (rank_version == kind::poppy)
          millis_pre = epic::gpu::launch_presearch_kernel<256, false, kind::poppy>(
              grid_size, block_size, dm);
        else if (rank_version == kind::cum_poppy)
          millis_pre = epic::gpu::launch_presearch_kernel<256, false, kind::cum_poppy>(
              grid_size, block_size,
              dm);
        break;
      case 512:
        if (rank_version == kind::poppy)
          millis_pre = epic::gpu::launch_presearch_kernel<512, false, kind::poppy>(
              grid_size, block_size,
              dm);
        else if (rank_version == kind::cum_poppy)
          millis_pre = epic::gpu::launch_presearch_kernel<512, false, kind::cum_poppy>(
              grid_size, block_size,
              dm);
        break;
      case 1024:
        if (parameters.with_shuffles)
        {
          if (rank_version == kind::poppy)
            millis_pre = epic::gpu::launch_presearch_kernel<1024, true, kind::poppy>(
                grid_size, block_size,
                dm);
          else if (rank_version == kind::cum_poppy)
            millis_pre = epic::gpu::launch_presearch_kernel<1024, true, kind::cum_poppy>(
                grid_size, block_size,
                dm);
        }
        else
        {
          if (rank_version == kind::poppy)
            millis_pre = epic::gpu::launch_presearch_kernel<1024, false, kind::poppy>(
                grid_size, block_size,
                dm);
          else if (rank_version == kind::cum_poppy)
            millis_pre = epic::gpu::launch_presearch_kernel<1024, false, kind::cum_poppy>(
                grid_size, block_size,
                dm);
        }
        break;
      case 2048:
        if (parameters.with_shuffles)
        {
          if (rank_version == kind::poppy)
            millis_pre = epic::gpu::launch_presearch_kernel<2048, true, kind::poppy>(
                grid_size, block_size,
                dm);
          else if (rank_version == kind::cum_poppy)
            millis_pre = epic::gpu::launch_presearch_kernel<2048, true, kind::cum_poppy>(
                grid_size, block_size,
                dm);
        }
        else
        {
          if (rank_version == kind::poppy)
            millis_pre = epic::gpu::launch_presearch_kernel<2048, false, kind::poppy>(
                grid_size, block_size,
                dm);
          else if (rank_version == kind::cum_poppy)
            millis_pre = epic::gpu::launch_presearch_kernel<2048, false, kind::cum_poppy>(
                grid_size, block_size,
                dm);
        }
        break;
      case 4096:
        if (rank_version == kind::poppy)
          millis_pre = epic::gpu::launch_presearch_kernel<4096, false, kind::poppy>(
              grid_size, block_size,
              dm);
        // Super block size 4096 with cum_poppy is not supported
        else if (rank_version == kind::cum_poppy)
          printf("Super block size 4096 with cumulative poppy is not supported. The search is done using the regular poppy.");
        millis_pre = epic::gpu::launch_presearch_kernel<4096, false, kind::poppy>(grid_size, block_size,
                                                                                  dm);

        break;
      }
    }
    return 0;
  }

}