#pragma once
#include "../globals.hpp"
#include "../kmer_string.hpp"
#include "cuda.hpp"
#include "device_memory.hpp"

namespace epic
{
  namespace gpu
  {

    struct DeviceMemoryQueries
    {

      u64 *kmer_string = nullptr;
      u64 *positions_and_results = nullptr;

      u64 size_kmer_string = 0ULL;
      u64 size_positions_and_results = 0ULL;

      int allocate_device_memory(epic::KmerString &, epic::gpu::DeviceMemory &);
      int send_to_device_memory_async(epic::KmerString &, epic::gpu::DeviceMemory &);

      DeviceMemoryQueries() = default;
      ~DeviceMemoryQueries();
    };

    DeviceMemoryQueries::~DeviceMemoryQueries()
    {
      if (kmer_string)
        deviceFree(kmer_string);
      if (positions_and_results)
        deviceFree(positions_and_results);
    }

    int DeviceMemoryQueries::allocate_device_memory(epic::KmerString &host_kmer_string, epic::gpu::DeviceMemory &dm)
    {
      size_kmer_string = host_kmer_string.number_of_words_in_kmer_string * sizeof(u64);
      size_positions_and_results = host_kmer_string.number_of_kmer_positions_with_padding * sizeof(u64);
      deviceError_t err1 = deviceMalloc((void **)&kmer_string, size_kmer_string);
      deviceError_t err2 = deviceMalloc((void **)&positions_and_results, size_positions_and_results);
      if (err1)
      {
        print_device_error(err1, "");
        return 1;
      }
      if (err2)
      {
        print_device_error(err2, "");
        return 1;
      }
      return 0;
    }

    int DeviceMemoryQueries::send_to_device_memory_async(epic::KmerString &host_kmer_string, epic::gpu::DeviceMemory &dm)
    {
      deviceMemcpyAsyncHostToDevice(kmer_string, host_kmer_string.kmer_string, size_kmer_string, dm.device_stream.stream);
      deviceMemcpyAsyncHostToDevice(positions_and_results, host_kmer_string.kmer_positions, size_positions_and_results, dm.device_stream.stream);
      return 0;
    }
  }
}
