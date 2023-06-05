#pragma once
#include "../enums.hpp"
#include "../epicseq/kmer_data.hpp"
#include "../globals.hpp"
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
        epic::KmerData &kmer_data)
    {
      DEBUG_CODE(
          fprintf(stderr, "In launch_search_kernel():\n");
          fprintf(stderr, "grid_size %" PRIu64 "\n", (u64)grid_size);
          fprintf(stderr, "block_size %" PRIu64 "\n", (u64)block_size);
          fprintf(stderr, "kmer_data.size_positions_and_results %" PRIu64 "\n", (u64)kmer_data.size_positions_and_results);

          fprintf(stderr, "dm.kmer_length %" PRIu64 "\n", (u64)dm.kmer_length);
          fprintf(stderr, "dm.size_A %" PRIu64 "\n", (u64)dm.size_A);
          fprintf(stderr, "dm.presearch_mer_length %" PRIu64 "\n", (u64)dm.presearch_mer_length);
          if (dm.node_lefts == nullptr)
              fprintf(stderr, "dm.node_lefts = nullptr");
          fprintf(stderr, "Before calling search kernel: ");
          epic::gpu::get_and_print_last_error();)

      dm.device_stream.start_timer();
      epic::gpu::search<superblock_size, shuffles, rank_version, with_presearch><<<dim3(grid_size), dim3(block_size), 0, dm.device_stream.stream>>>(
          kmer_data.d_kmer_string,
          dm.counts_before,
          dm.SBWT,
          dm.L0,
          dm.L12,
          dm.node_lefts,
          dm.node_rights,
          dm.kmer_length,
          kmer_data.d_positions_and_results,
          dm.presearch_mer_length);
      dm.device_stream.stop_timer();
      float millis = dm.device_stream.duration_in_millis();
      DEBUG_CODE(
          fprintf(stderr, "After calling search kernel: ");
          epic::gpu::get_and_print_last_error(); // To debug.
          fprintf(stderr, "Search takes (printed inside launch_search_kernel()): %f ms\n", millis);)
      return millis;
    }

    template <int superblock_size, bool shuffles, int rank_version>
    inline float launch_presearch_kernel(
        std::size_t grid_size,
        std::size_t block_size,
        epic::gpu::DeviceMemory &dm)
    {
      DEBUG_CODE(
          fprintf(stderr, "Last error before and after presearch kernel call:\n");
          epic::gpu::get_and_print_last_error();)

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
      DEBUG_CODE(epic::gpu::get_and_print_last_error();)

      return dm.device_stream.duration_in_millis();
    }

  }
}
