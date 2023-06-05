#pragma once
#include "../globals.hpp"
#include "cuda.hpp"
#include "device_stream.hpp"

typedef uint64_t u64;
typedef uint32_t u32;

namespace epic
{
  namespace gpu
  {

    struct DeviceMemory
    {
      u64 *A = nullptr;
      u64 *C = nullptr;
      u64 *G = nullptr;
      u64 *T = nullptr;

      u64 *counts_before = nullptr;

      u64 *L0_A = nullptr;
      u64 *L12_A = nullptr;
      u64 *L0_C = nullptr;
      u64 *L12_C = nullptr;
      u64 *L0_G = nullptr;
      u64 *L12_G = nullptr;
      u64 *L0_T = nullptr;
      u64 *L12_T = nullptr;

      u64 **SBWT = nullptr;
      u64 **L0 = nullptr;
      u64 **L12 = nullptr;

      u64 *node_lefts = nullptr;
      u64 *node_rights = nullptr;

      u64 size_A = 0ULL; // Sizes are in bytes;
      u64 size_C = 0ULL;
      u64 size_G = 0ULL;
      u64 size_T = 0ULL;

      u64 size_counts_before = 5 * sizeof(u64);
      u64 size_L0_A = 0ULL;
      u64 size_L12_A = 0ULL;
      u64 size_L0_C = 0ULL;
      u64 size_L12_C = 0ULL;
      u64 size_L0_G = 0ULL;
      u64 size_L12_G = 0ULL;
      u64 size_L0_T = 0ULL;
      u64 size_L12_T = 0ULL;

      u64 size_SBWT = 4 * sizeof(u64 *);
      u64 size_L0 = 4 * sizeof(u64 *);
      u64 size_L12 = 4 * sizeof(u64 *);

      u64 size_node_lefts = 0ULL;
      u64 size_node_rights = 0ULL;

      u32 kmer_length = 0ULL;
      u32 presearch_mer_length = 0ULL;

      DeviceStream device_stream;

      void set_sizes_of_bit_vectors(u64);
      void set_sizes_of_rank_data_structures(u64, u64); // size of L0, size of L12
      void set_sizes_of_node_lefts_and_rights(u64);
      void set_lengths_of_kmer_and_presearch_mer(u32, u32);
      int allocate_device_memory();

      DeviceMemory();
      ~DeviceMemory();
    };

    DeviceMemory::DeviceMemory()
    {
      device_stream.create();
    }

    DeviceMemory::~DeviceMemory()
    {
      DEBUG_BEFORE_DESTRUCT("DeviceMemory (ALL)");
      if (A)
        cudaFree(A);
      if (C)
        cudaFree(C);
      if (G)
        cudaFree(G);
      if (T)
        cudaFree(T);
      if (counts_before)
        cudaFree(counts_before);
      if (L0_A)
        cudaFree(L0_A);
      if (L12_A)
        cudaFree(L12_A);
      if (L0_C)
        cudaFree(L0_C);
      if (L12_C)
        cudaFree(L12_C);
      if (L0_G)
        cudaFree(L0_G);
      if (L12_G)
        cudaFree(L12_G);
      if (L0_T)
        cudaFree(L0_T);
      if (L12_T)
        cudaFree(L12_T);
      if (SBWT)
        cudaFree(SBWT);
      if (L0)
        cudaFree(L0);
      if (L12)
        cudaFree(L12);
      if (node_lefts)
        cudaFree(node_lefts);
      if (node_rights)
        cudaFree(node_rights);
      DEBUG_AFTER_DESTRUCT("DeviceMemory (ALL)");
    }

    // Number of uint64_t words as a parameter.
    void DeviceMemory::set_sizes_of_bit_vectors(u64 number_of_words)
    {
      size_A = size_C = size_G = size_T = (number_of_words * sizeof(u64));
    }

    // Number of uint64_t words of L0 and L12 as parameters.
    void DeviceMemory::set_sizes_of_rank_data_structures(u64 number_of_words_L0, u64 number_of_words_L12)
    {
      size_L0_A = size_L0_C = size_L0_G = size_L0_T = (number_of_words_L0 * sizeof(u64));
      size_L12_A = size_L12_C = size_L12_G = size_L12_T = (number_of_words_L12 * sizeof(u64));
    }
    void DeviceMemory::set_sizes_of_node_lefts_and_rights(u64 number_of_words)
    {
      size_node_lefts = size_node_rights = number_of_words * sizeof(u64);
    }

    void DeviceMemory::set_lengths_of_kmer_and_presearch_mer(u32 t_kmer_length, u32 t_presearch_mer_length)
    {
      kmer_length = t_kmer_length;
      presearch_mer_length = t_presearch_mer_length;
    }

    int DeviceMemory::allocate_device_memory()
    {
      deviceError_t errors[18];

      errors[0] = deviceMalloc((void **)&A, size_A);
      errors[1] = deviceMalloc((void **)&C, size_C);
      errors[2] = deviceMalloc((void **)&G, size_G);
      errors[3] = deviceMalloc((void **)&T, size_T);
      errors[4] = deviceMalloc((void **)&counts_before, size_counts_before);

      errors[5] = deviceMalloc((void **)&L0_A, size_L0_A);
      errors[6] = deviceMalloc((void **)&L0_C, size_L0_C);
      errors[7] = deviceMalloc((void **)&L0_G, size_L0_G);
      errors[8] = deviceMalloc((void **)&L0_T, size_L0_T);
      errors[9] = deviceMalloc((void **)&L12_A, size_L12_A);
      errors[10] = deviceMalloc((void **)&L12_C, size_L12_C);
      errors[11] = deviceMalloc((void **)&L12_G, size_L12_G);
      errors[12] = deviceMalloc((void **)&L12_T, size_L12_T);

      errors[13] = deviceMalloc((void ***)&SBWT, size_SBWT);
      errors[14] = deviceMalloc((void ***)&L0, size_L0);
      errors[15] = deviceMalloc((void ***)&L12, size_L12);

      errors[16] = deviceMalloc((void **)&node_lefts, size_node_lefts);
      errors[17] = deviceMalloc((void **)&node_rights, size_node_rights);

      int number_of_errors = 0;
      for (int i = 0; i < 18; i++)
      {
        if (errors[i])
        {
          DEBUG_CODE(print_device_error(errors[i], "");)
          number_of_errors++;
        }
      }
      if (number_of_errors)
        fprintf(stderr, "Allocating device memory failed.\n");
      return number_of_errors;
    }

  }
}