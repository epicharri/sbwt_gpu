#pragma once
#include "../enums.hpp"
#include "../globals.hpp"
#include "../kmer_string.hpp"
#include "cuda.hpp"
#include "device_memory.hpp"
#include "device_stream.hpp"
#include "presearch_kernel.hpp"
#include "search_kernel.hpp"
#include <cstdlib>

namespace epic
{
  namespace gpu
  {

    template <int superblock_size,
              bool shuffles = false,
              int rank_version = epic::kind::poppy,
              bool with_presearch = true>
    inline float launch_search_kernel(
        std::size_t grid_size,
        std::size_t block_size,
        epic::gpu::DeviceMemory &dm,
        epic::gpu::DeviceMemoryQueries &dm_queries,
        epic::KmerString &kmer_string)
    {
      if (kmer_string.use_unified_memory_for_queries)
      {

        epic::gpu::DeviceStream ds1;
        epic::gpu::DeviceStream ds2;
        ds1.create();
        ds2.create();
        dm.device_stream.start_timer();
        int gpu_id = -1;
        epic::gpu::print_device_error(cudaGetDevice(&gpu_id), "");

        epic::gpu::print_device_error(cudaMemPrefetchAsync(kmer_string.kmer_string, kmer_string.size_kmer_string, gpu_id, ds2.stream), "");
        epic::gpu::get_and_print_last_error();

        epic::gpu::print_device_error(cudaMemPrefetchAsync(kmer_string.search_results, kmer_string.size_kmer_positions, gpu_id, ds1.stream), "");
        epic::gpu::get_and_print_last_error();

        epic::gpu::search_unified<superblock_size, shuffles, rank_version, with_presearch><<<dim3(grid_size), dim3(block_size), 0, dm.device_stream.stream>>>(
            kmer_string.kmer_string,
            dm.counts_before,
            dm.SBWT,
            dm.L0,
            dm.L12,
            dm.node_lefts,
            dm.node_rights,
            dm.kmer_length,
            kmer_string.search_results,
            dm.presearch_mer_length);

        // Syncronizing the device should not be needed here, because in ...duration_in_millis()
        //   there is cudaEventSynchronice.
        epic::gpu::get_and_print_last_error();

        epic::gpu::print_device_error(cudaDeviceSynchronize(), "");

        dm.device_stream.stop_timer();
        float time_in_millis = dm.device_stream.duration_in_millis();
        epic::gpu::get_and_print_last_error();
        return time_in_millis;
      }
      else
      {
        dm.device_stream.start_timer();
        epic::gpu::search<superblock_size, shuffles, rank_version, with_presearch><<<dim3(grid_size), dim3(block_size), 0, dm.device_stream.stream>>>(
            dm_queries.kmer_string,
            dm.counts_before,
            dm.SBWT,
            dm.L0,
            dm.L12,
            dm.node_lefts,
            dm.node_rights,
            dm.kmer_length,
            dm_queries.positions_and_results,
            dm.presearch_mer_length);
        dm.device_stream.stop_timer();
        return dm.device_stream.duration_in_millis();
      }
    }

    template <int superblock_size, bool shuffles, int rank_version>
    inline float launch_presearch_kernel(
        std::size_t grid_size,
        std::size_t block_size,
        epic::gpu::DeviceMemory &dm)
    {
      dm.device_stream.start_timer();
      epic::gpu::presearch<superblock_size, shuffles, rank_version><<<dim3(grid_size), dim3(block_size), 0, dm.device_stream.stream>>>(
          dm.counts_before,
          dm.SBWT,
          dm.L0,
          dm.L12,
          dm.node_lefts,
          dm.node_rights,
          dm.presearch_mer_length);
      dm.device_stream.stop_timer();
      return dm.device_stream.duration_in_millis();
    }

  }
}
