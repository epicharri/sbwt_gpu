#pragma once
#include "../globals.hpp"
#include "../gpu/cuda.hpp"
#include "../utils/helpers.hpp"

namespace epic
{

  struct KmerData
  {
    u64 *kmer_string = nullptr;
    u64 *positions_and_results = nullptr;
    u64 *d_kmer_string = nullptr;
    u64 *d_positions_and_results = nullptr;

    u8 *d_raw_data = nullptr;
    u64 size_raw_data = 0ULL;
    u64 size_kmer_string = 0ULL;           // In bytes
    u64 size_positions_and_results = 0ULL; // In bytes
    u64 number_of_words_in_kmer_string = 0ULL;
    u64 number_of_positions = 0ULL;
    u64 number_of_positions_padded = 0ULL;
    int allocate_memory(epic::gpu::DeviceStream &);
    KmerData() = default;
    ~KmerData();
  };

  KmerData::~KmerData()
  {
    DEBUG_BEFORE_DESTRUCT("KmerData.kmer_string");
    if (kmer_string)
      delete[] kmer_string;
    DEBUG_AFTER_DESTRUCT("KmerData.kmer_string");

    DEBUG_BEFORE_DESTRUCT("KmerData.positions_and_results");
    if (positions_and_results)
      delete[] positions_and_results;
    DEBUG_AFTER_DESTRUCT("KmerData.positions_and_results");

    DEBUG_BEFORE_DESTRUCT("KmerData.d_kmer_string");
    if (d_kmer_string)
      cudaFree(d_kmer_string);
    DEBUG_AFTER_DESTRUCT("KmerData.d_kmer_string");

    DEBUG_BEFORE_DESTRUCT("KmerData.d_positions_and_results");
    if (d_positions_and_results)
      cudaFree(d_positions_and_results);
    DEBUG_AFTER_DESTRUCT("KmerData.d_positions_and_results");

    DEBUG_BEFORE_DESTRUCT("KmerData.d_raw_data");
    if (d_raw_data)
      cudaFree(d_raw_data);
    DEBUG_AFTER_DESTRUCT("KmerData.d_raw_data");
  }

  int KmerData::allocate_memory(epic::gpu::DeviceStream &ds)
  {

    if (epic::utils::allocate_host_memory(kmer_string, size_kmer_string, "Allocate memory for a kmer string."))
      return 1;
    if (epic::utils::allocate_host_memory(positions_and_results, size_positions_and_results, "Allocate memory for the positions and results."))
      return 1;

    CHECK(cudaMallocAsync((void **)&d_kmer_string, size_kmer_string, ds.stream));
    CHECK(cudaMallocAsync((void **)&d_positions_and_results, size_positions_and_results, ds.stream));
    number_of_positions_padded = size_positions_and_results / sizeof(u64);
    CHECK(cudaMallocAsync((void **)&d_raw_data, size_raw_data, ds.stream));
    return 0;
  }

}