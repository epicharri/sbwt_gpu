#pragma once
#include "../include/bit_vector.hpp"
#include "../include/globals.hpp"
#include "../include/gpu/cuda.hpp"
#include "../include/gpu/device_stream.hpp"
#include "../include/parameters.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <string>
#include <vector>

namespace epic
{

  struct KmerString
  {
    u64 *kmer_positions = nullptr;
    u64 *kmer_string = nullptr;
    u64 *search_results = nullptr;

    bool use_unified_memory_for_queries = false;
    u32 kmer_size = 0U;
    u64 number_of_kmer_positions = 0ULL;
    u64 number_of_kmer_positions_with_padding = 0ULL;
    u64 number_of_letters_in_kmer_string = 0ULL;
    u64 number_of_words_in_kmer_string = 0ULL;

    u64 size_kmer_positions = 0ULL; // Size in bytes, with padding.
    u64 size_kmer_string = 0ULL;    // Size in bytes.

    bool use_unified_memory = false;

    std::vector<std::string> textlines;
    int construct_kmer_string_and_fill_positions();
    u64 round_up_first_to_multiple_of_second(u64 t_number, u64 t_coefficient);
    void set_kmer_size(u32);

    void set_unified_memory_flag(bool);

    int allocate_unified_memory();
    int allocate_host_memory();
    int allocate_memory();
    int prefetch_data_async(epic::gpu::DeviceStream &);

    int read_kmers(std::string filename);

    KmerString() = default;
    ~KmerString();
  };

  KmerString::~KmerString()
  {
    if (use_unified_memory_for_queries)
    {
      if (kmer_positions)
        delete[] kmer_positions;
      //      cudaFree(kmer_positions);
      if (kmer_string)
        cudaFree(kmer_string);
      if (search_results)
        cudaFree(search_results);
    }
    else
    {
      if (kmer_positions)
        delete[] kmer_positions;
      if (kmer_string)
        delete[] kmer_string;
      if (search_results)
        delete[] search_results;
    }
  }

  void KmerString::set_kmer_size(u32 t_kmer_size)
  {
    kmer_size = t_kmer_size;
  }

  void KmerString::set_unified_memory_flag(bool t_use_unified_memory_for_queries)
  {
    use_unified_memory_for_queries = t_use_unified_memory_for_queries;
  }

  inline u64 KmerString::round_up_first_to_multiple_of_second(u64 t_number, u64 t_coefficient)
  {
    return ((t_number + t_coefficient - 1ULL) / t_coefficient) * t_coefficient;
  }

  int KmerString::read_kmers(std::string filename)
  {
    number_of_kmer_positions = 0ULL;
    number_of_letters_in_kmer_string = 0ULL;
    std::string textline;
    std::ifstream textfile(filename);
    if (!textfile.is_open())
    {
      fprintf(stderr, "Text file %s could not be opened.", filename);
      return 1;
    }
    while (std::getline(textfile, textline))
    {
      if (textline[0] != '>')
      {
        u32 length_of_line = textline.length();
        textlines.push_back(textline);
        number_of_kmer_positions += length_of_line - kmer_size + 1U; // e.g. length=100, k=31, positions 0..69 => 70 positions.
        number_of_letters_in_kmer_string += length_of_line;
      }
    }
    textfile.close();
    return 0;
  }

  inline int KmerString::allocate_memory()
  {
    if (use_unified_memory_for_queries)
      return allocate_unified_memory();
    return allocate_host_memory();
  }

  inline int KmerString::allocate_host_memory()
  {
    kmer_positions = new u64[number_of_kmer_positions_with_padding];
    kmer_string = new u64[number_of_words_in_kmer_string];
    search_results = new u64[number_of_kmer_positions_with_padding];
    return 0;
  }

  inline int KmerString::allocate_unified_memory()
  {

    BENCHMARK_CODE(
        std::size_t free_memory = 0;
        std::size_t total_memory = 0;
        cudaMemGetInfo(&free_memory, &total_memory);
        fprintf(stderr, "Free memory: %zu bytes.\n", free_memory);
        fprintf(stderr, "Total memory: %zu bytes.\n", total_memory);)

    kmer_positions = new u64[size_kmer_positions];
    CHECK(cudaMallocManaged(&kmer_string, sizeof(u64) * number_of_words_in_kmer_string));
    CHECK(cudaMallocManaged(&search_results, sizeof(u64) * number_of_kmer_positions_with_padding));
    return 0;
  }

  inline int KmerString::prefetch_data_async(epic::gpu::DeviceStream &ds)
  {
    int device = -1;
    cudaGetDevice(&device);
    CHECK(cudaMemPrefetchAsync(kmer_positions, sizeof(u64) * number_of_kmer_positions_with_padding, device, ds.stream));
    CHECK(cudaMemPrefetchAsync(kmer_string, sizeof(u64) * number_of_words_in_kmer_string, device, ds.stream));
    return 0;
  }

  int KmerString::construct_kmer_string_and_fill_positions()
  {

    number_of_kmer_positions_with_padding = round_up_first_to_multiple_of_second(number_of_kmer_positions, 64ULL);
    number_of_words_in_kmer_string = round_up_first_to_multiple_of_second(number_of_letters_in_kmer_string, 32ULL) / 32ULL;

    size_kmer_positions = sizeof(u64) * number_of_kmer_positions_with_padding;
    size_kmer_string = sizeof(u64) * number_of_words_in_kmer_string;

    if (allocate_memory())
      return 1;

    u64 bits[256];
    bits['A'] = 0ULL;
    bits['C'] = 1ULL;
    bits['G'] = 2ULL;
    bits['T'] = 3ULL;

    for (u64 i = number_of_kmer_positions; i < number_of_kmer_positions_with_padding; i += 1ULL)
    {
      kmer_positions[i] = 0ULL;
      search_results[i] = 0ULL; // This can be done in other way with no need to store values to both arrays. Now for benchmarkin purposes this way.
    }

    u64 i_kmer_positions = 0ULL;
    u64 i_kmer_string = 0ULL;
    for (u64 i = 0; i < textlines.size(); i += 1ULL)
    {
      std::string row = textlines[i];
      u32 positionsInRow = row.length() - kmer_size + 1;

      for (u32 j = 0; j < positionsInRow; j += 1U)
      {

        kmer_positions[i_kmer_positions] = i_kmer_string;
        search_results[i_kmer_positions] = i_kmer_string; // This can be done in other way with no need to store values to both arrays. Now for benchmarkin purposes this way.

        u64 acgt = bits[row[j]] << ((31ULL - (i_kmer_string & 31ULL)) << 1);
        if ((i_kmer_string & 31ULL) == 0ULL)
        {
          kmer_string[(i_kmer_string >> 5)] = 0ULL;
        }
        kmer_string[(i_kmer_string >> 5)] = kmer_string[(i_kmer_string >> 5)] | acgt;
        i_kmer_positions += 1ULL;
        i_kmer_string += 1ULL;
      }

      for (u32 j = positionsInRow; j < row.length(); j += 1ULL)
      {
        u64 acgt = bits[row[j]] << ((31ULL - (i_kmer_string & 31ULL)) << 1);
        if ((i_kmer_string & 31ULL) == 0ULL)
        {
          kmer_string[(i_kmer_string >> 5)] = 0ULL;
        }
        kmer_string[(i_kmer_string >> 5)] = kmer_string[(i_kmer_string >> 5)] | acgt;
        i_kmer_string += 1ULL;
      }
    }
    textlines.clear();
    textlines.shrink_to_fit();
    return 0;
  }

}