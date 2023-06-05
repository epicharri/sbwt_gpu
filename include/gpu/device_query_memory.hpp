#pragma once

#include "../globals.hpp"
#include "cuda.hpp"

namespace epic
{
  namespace gpu
  {
    class DeviceQueryMemory
    {
    public:
      u64 *positions_and_results = nullptr;
      u64 *nucleobase_string = nullptr;
      u64 *d_positions_and_results = nullptr;
      u64 *d_nucleobase_string = nullptr;
      u64 size_of_positions_with_padding = 0ULL;
      u64 size_of_nucleobase_string_with_padding = 0ULL;
      u32 kmer_size = 0U;
      bool use_unified_memory = false;
      int allocate_memory();
      int allocate_unified_memory();
      int allocate_host_and_device_memory();
      void set_size_of_positions_with_padding(u64);
      void set_size_of_nucleobase_string_with_padding(u64);
      void set_kmer_size(u32);
      void set_unified_memory_flag(bool);

      DeviceQueryMemory() = default;
      ~DeviceQueryMemory();
    };

    DeviceQueryMemory::DeviceQueryMemory()
    {
    }

    DeviceQueryMemory::~DeviceQueryMemory()
    {
      if (use_unified_memory)
      {
        if (nucleobase_string)
          cudaFree(nucleobase_string);
        if (positions_and_results)
          cudaFree(positions_and_results);
      }
      else
      {
        if (nucleobase_string)
          delete[] nucleobase_string;
        if (positions_and_results)
          delete[] positions_and_results;
        if (d_nucleobase_string)
          cudaFree(d_nucleobase_string);
        if (d_positions_and_results)
          cudaFree(d_positions_and_results);
      }
    }

    int DeviceQueryMemory::KmerString::allocate_memory()
    {
      if (use_unified_memory)
        return allocate_unified_memory();
      return allocate_host_and_global_memory();
    }

    int DeviceQueryMemory::allocate_unified_memory()
    {
      CHECK(cudaMallocManaged(&nucleobase_string, size_of_nucleobase_string_with_padding));
      CHECK(cudaMallocManaged(&positions_and_results, size_of_positions_with_padding));
      d_nucleobase_string = nucleobase_string;
      d_positions_and_results = positions_and_results;
      return 0;
    }

    int DeviceQueryMemory::allocate_host_and_device_memory()
    {
      try
      {
        nucleobase_string = new u64[size_of_nucleobase_string_with_padding];
        positions_and_results = new u64[size_of_positions_with_padding];
      }
      catch (const std::bad_alloc &e)
      {
        std::cerr << e.what() << '\n';
      }
      deviceError_t err1 = deviceMalloc((void **)&nucleobase_string, size_nucleobase_string_with_padding);
      deviceError_t err2 = deviceMalloc((void **)&positions_and_results, size_of_positions);
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

    void DeviceQueryMemory::set_kmer_size(u32 t_kmer_size)
    {
      kmer_size = t_kmer_size;
    }

    void DeviceQueryMemory::set_unified_memory_flag(
        bool t_use_unified_memory)
    {
      use_unified_memory = t_use_unified_memory;
    }

    void DeviceQueryMemory::set_size_of_positions_with_padding(u64 bytes)
    {
      size_of_positions_with_padding = bytes;
    }

    void DeviceQueryMemory::set_size_of_nucleobase_string_with_padding(u64 bytes)
    {
      size_of_nucleobase_string_with_padding = bytes;
    }

  }
}