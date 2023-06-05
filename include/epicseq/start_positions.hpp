#pragma once
#include "../epicseq/kmer_data.hpp"
#include "../globals.hpp"
#include "../gpu/cuda.hpp"
#include "../gpu/positions_kernel.hpp"
#include "../utils/helpers.hpp"
#include <vector>

namespace epic
{

  class DeviceStartPositions
  {
  public:
    u64 *d_start_positions = nullptr;
    float millis_create_positions = 0.0;
    int create_positions(epic::gpu::DeviceStream &, std::vector<int64> &, epic::KmerData &, u32);
    DeviceStartPositions() = default;
    ~DeviceStartPositions();
  };

  DeviceStartPositions::~DeviceStartPositions()
  {
    DEBUG_BEFORE_DESTRUCT("DeviceStartPositions.d_start_positions");
    if (d_start_positions)
      cudaFree(d_start_positions);
    DEBUG_AFTER_DESTRUCT("DeviceStartPositions.d_start_positions");
  }

  int DeviceStartPositions::create_positions(
      epic::gpu::DeviceStream &ds,
      std::vector<int64> &start_positions,
      epic::KmerData &kmer_data,
      u32 kmer_size)
  {
    ds.start_timer();
    u64 number_of_start_positions = start_positions.size() - 1;
    u64 size_start_positions = sizeof(u64) * (number_of_start_positions + 1);
    fprintf(stderr, "size_start_positions: %" PRIu64 " in int DeviceStartPositions::create_positions()\n", size_start_positions);
    CHECK(cudaMallocAsync((void **)&d_start_positions, size_start_positions, ds.stream));
    CHECK(cudaMemcpyAsync(d_start_positions, start_positions.data(), size_start_positions, cudaMemcpyHostToDevice, ds.stream));
    u64 block_size, grid_size;
    block_size = 256;
    grid_size = epic::utils::round_up_first_to_multiple_of_second<u64>(number_of_start_positions, block_size) / block_size;

    fprintf(stderr, "Last error before and after create_positions kernel call:\n");
    epic::gpu::get_and_print_last_error();

    epic::gpu::create_positions<<<dim3(grid_size), dim3(block_size), 0, ds.stream>>>(
        d_start_positions,
        kmer_data.d_positions_and_results,
        number_of_start_positions,
        kmer_size);

    ds.stop_timer();
    millis_create_positions = ds.duration_in_millis();

    epic::gpu::get_and_print_last_error();

    // Debugging starts
    /*
    CHECK(cudaMemcpyAsync(kmer_data.positions_and_results, kmer_data.d_positions_and_results, kmer_data.size_positions_and_results, cudaMemcpyDeviceToHost, ds.stream));
    cudaDeviceSynchronize();
    for (int i = 0; i < 1000; i++)
    {
      fprintf(stderr, "%d: %" PRIu64 "\n", kmer_data.positions_and_results[i]);
    }
*/
    // Debugging ends

    //    CHECK(cudaFreeAsync(d_start_positions, ds.stream));
    return 0;
  }

}