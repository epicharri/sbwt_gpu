#pragma once
#include "../include/bit_vector.hpp"
#include "../include/compare_to_answers.hpp"
#include "../include/enums.hpp"
#include "../include/globals.hpp"
#include "../include/gpu/device_memory.hpp"
#include "../include/gpu/device_memory_queries.hpp"
#include "../include/gpu/device_stream.hpp"
#include "../include/gpu/kernel_launchers.hpp"
#include "../include/gpu/search_kernel.hpp"
#include "../include/kmer_string.hpp"
#include "../include/parameters.hpp"
#include "../include/presearch.hpp"
#include "../include/rank_data_structures.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace epic
{

  struct Search
  {
    float millis_send_to_device = 0.0;
    float millis = 0.0;
    float millis_copy_results = 0.0;
    float millis_pre = 0.0;
    float millis_read_kmers = 0.0;
    float millis_construct_kmer_string = 0.0;
    float millis_send_search_copy_results = 0.0;
    bool use_unified_memory_for_queries = false;
    int search(epic::Parameters &, epic::gpu::DeviceMemory &);
    int search_in_gpu(epic::Parameters &, epic::gpu::DeviceMemory &, epic::gpu::DeviceMemoryQueries &, epic::KmerString &);
    int search_using_unified_memory_for_queries(
        epic::KmerString &,
        epic::Parameters &,
        epic::gpu::DeviceMemory &,
        epic::gpu::DeviceMemoryQueries &);
    int search_using_gpu_memory_for_queries(
        epic::KmerString &,
        epic::Parameters &,
        epic::gpu::DeviceMemory &,
        epic::gpu::DeviceMemoryQueries &);
    void print_benchmark_information(epic::KmerString &, epic::Parameters &, epic::gpu::DeviceMemory &);
    void set_millis_pre(float);
    Search() = default;
  };

  // Search using unified memory
  int Search::search_using_unified_memory_for_queries(
      epic::KmerString &kmer_string,
      epic::Parameters &parameters,
      epic::gpu::DeviceMemory &dm,
      epic::gpu::DeviceMemoryQueries &dm_queries)
  {

    auto read_kmers_start = START_TIME;
    if (kmer_string.read_kmers(parameters.fileQueries))
      return 1;
    auto read_kmers_stop = STOP_TIME;
    millis_read_kmers = DURATION_IN_MILLISECONDS(read_kmers_start, read_kmers_stop);
    auto construct_kmer_string_start = START_TIME;
    if (kmer_string.construct_kmer_string_and_fill_positions())
      return 1;
    auto construct_kmer_string_stop = STOP_TIME;
    millis_construct_kmer_string = DURATION_IN_MILLISECONDS(construct_kmer_string_start, construct_kmer_string_stop);

    if (search_in_gpu(parameters, dm, dm_queries, kmer_string))
    {
      fprintf(stderr, "Search in GPU did not succeed.\n");
      return 1;
    }

    millis_send_search_copy_results = millis;
    return 0;
  }

  int Search::search_using_gpu_memory_for_queries(
      epic::KmerString &kmer_string,
      epic::Parameters &parameters,
      epic::gpu::DeviceMemory &dm,
      epic::gpu::DeviceMemoryQueries &dm_queries)
  {

    auto read_kmers_start = START_TIME;
    if (kmer_string.read_kmers(parameters.fileQueries))
      return 1;
    auto read_kmers_stop = STOP_TIME;
    millis_read_kmers = DURATION_IN_MILLISECONDS(read_kmers_start, read_kmers_stop);
    auto construct_kmer_string_start = START_TIME;
    if (kmer_string.construct_kmer_string_and_fill_positions())
      return 1;
    auto construct_kmer_string_stop = STOP_TIME;
    millis_construct_kmer_string = DURATION_IN_MILLISECONDS(construct_kmer_string_start, construct_kmer_string_stop);
    dm.device_stream.start_timer();
    dm_queries.allocate_device_memory(kmer_string, dm);
    dm_queries.send_to_device_memory_async(kmer_string, dm);
    dm.device_stream.stop_timer();
    millis_send_to_device = dm.device_stream.duration_in_millis();

    if (search_in_gpu(parameters, dm, dm_queries, kmer_string))
    {
      fprintf(stderr, "Search in GPU did not succeed.\n");
      return 1;
    }

    dm.device_stream.start_timer();
    cudaMemcpy(kmer_string.search_results, dm_queries.positions_and_results, kmer_string.number_of_kmer_positions * sizeof(u64), cudaMemcpyDeviceToHost);
    dm.device_stream.stop_timer();
    millis_copy_results = dm.device_stream.duration_in_millis();
    millis_send_search_copy_results = millis_send_to_device + millis + millis_copy_results;
    return 0;
  }

  int Search::search(epic::Parameters &parameters, epic::gpu::DeviceMemory &dm)
  {
    use_unified_memory_for_queries = parameters.use_unified_memory_for_queries;
    epic::gpu::DeviceMemoryQueries dm_queries;

    epic::KmerString kmer_string;
    kmer_string.set_kmer_size(parameters.k);
    kmer_string.set_unified_memory_flag(use_unified_memory_for_queries);

    if (use_unified_memory_for_queries)
    {
      if (search_using_unified_memory_for_queries(kmer_string, parameters, dm, dm_queries))
      {
        fprintf(stderr, "Reading the queries and searching did not succeed.\n");
        return 1;
      }
    }

    if (!use_unified_memory_for_queries)
    {
      if (search_using_gpu_memory_for_queries(kmer_string, parameters, dm, dm_queries))
      {
        fprintf(stderr, "Reading the queries and searching did not succeed.\n");
        return 1;
      }
    }
    // This is to test if queries are correct. In production there must be saving the results to a file.
    epic::CompareToAnswers tester;
    tester.check(kmer_string, parameters.fileAnswers);
    print_benchmark_information(kmer_string, parameters, dm);
    return 0;
  }

  int Search::search_in_gpu(epic::Parameters &parameters, epic::gpu::DeviceMemory &dm, epic::gpu::DeviceMemoryQueries &dm_queries, epic::KmerString &kmer_string)
  {

    u64 block_size, grid_size;
    block_size = parameters.threads_per_block;
    grid_size = (kmer_string.number_of_kmer_positions_with_padding + block_size - 1ULL) / block_size;

    millis = 0.0;
    millis_copy_results = 0.0;
    int rank_version = parameters.rank_structure_version;

    if (DEVICE_IS_NVIDIA_A100)
    {
      cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, parameters.device_set_limit_search);
    }
    cudaDeviceSynchronize();
    if (parameters.with_presearch)
    {
      switch (parameters.bits_in_superblock)
      {
      case 256:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<256, false, epic::kind::poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        else if (rank_version == epic::kind::cum_poppy)
          millis = epic::gpu::launch_search_kernel<256, false, epic::kind::cum_poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        break;
      case 512:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<512, false, epic::kind::poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        else if (rank_version == epic::kind::cum_poppy)
          millis = epic::gpu::launch_search_kernel<512, false, epic::kind::cum_poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        break;
      case 1024:
        if (parameters.with_shuffles)
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<1024, true, epic::kind::poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<1024, true, epic::kind::cum_poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        }
        else
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<1024, false, epic::kind::poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<1024, false, epic::kind::cum_poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        }
        break;
      case 2048:
        if (parameters.with_shuffles)
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<2048, true, epic::kind::poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<2048, true, epic::kind::cum_poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        }
        else
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<2048, false, epic::kind::poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<2048, false, epic::kind::cum_poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        }
        break;
      case 4096:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<4096, false, epic::kind::poppy, true>(grid_size, block_size, dm, dm_queries, kmer_string);
        // Only poppy is supported with the superblock size 4096.
        break;
      }
    }
    else
    {
      switch (parameters.bits_in_superblock)
      {
      case 256:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<256, false, epic::kind::poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        else if (rank_version == epic::kind::cum_poppy)
          millis = epic::gpu::launch_search_kernel<256, false, epic::kind::cum_poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        break;
      case 512:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<512, false, epic::kind::poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        else if (rank_version == epic::kind::cum_poppy)
          millis = epic::gpu::launch_search_kernel<512, false, epic::kind::cum_poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        break;
      case 1024:
        if (parameters.with_shuffles)
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<1024, true, epic::kind::poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<1024, true, epic::kind::cum_poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        }
        else
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<1024, false, epic::kind::poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<1024, false, epic::kind::cum_poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        }
        break;
      case 2048:
        if (parameters.with_shuffles)
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<2048, true, epic::kind::poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<2048, true, epic::kind::cum_poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        }
        else
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<2048, false, epic::kind::poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<2048, false, epic::kind::cum_poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        }
        break;
      case 4096:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<4096, false, epic::kind::poppy, false>(grid_size, block_size, dm, dm_queries, kmer_string);
        // Only poppy is supported with the superblock size 4096.
        break;
      }
    }
    return 0;
  }

  void Search::set_millis_pre(float t_millis_pre)
  {
    millis_pre = t_millis_pre;
  }

  void Search::print_benchmark_information(epic::KmerString &kmer_string, epic::Parameters &parameters, epic::gpu::DeviceMemory &dm)
  {

    std::cerr << "Read queries from the file: " << millis_read_kmers << " ms\n";
    std::cerr << "Fill kmer string and positions: " << millis_construct_kmer_string << " ms\n";
    std::cerr << "\n";
    std::cerr << "Copy kmer string and search positions to the GPU: " << millis_send_to_device << " ms\n";
    std::cerr << "Search: " << millis << " milliseconds for " << kmer_string.number_of_kmer_positions << " queries.  \n";
    std::cerr << "Copy the results: " << millis_copy_results << " ms\n";

    if (parameters.use_unified_memory_for_queries)
      std::cerr << "Using Unified memory for the queries.\n";

    std::cerr << "Send the kmerstring and positions, search and copy the results TOTAL: " << millis_send_search_copy_results << " ms.\n";

    float nanosecondsperquery, nanosecondsperquery_pre, ns_per_query_total, millis_total, ns_per_query_real_total;
    millis_total = millis_read_kmers + millis_construct_kmer_string + millis_send_to_device + millis + millis_copy_results;
    ns_per_query_real_total = (float)((((double)millis_total) * 1000000.0) / (double)kmer_string.number_of_kmer_positions);
    ns_per_query_total = (float)((double)millis_send_search_copy_results * 1000000.0) / (double)kmer_string.number_of_kmer_positions;
    nanosecondsperquery = (float)((double)millis * 1000000.0) / (double)kmer_string.number_of_kmer_positions;
    nanosecondsperquery_pre = (float)((double)millis_pre * 1000000.0) / (double)kmer_string.number_of_kmer_positions;
    std::cerr << "Presearch: " << millis_pre << " milliseconds. \n";
    std::cerr << "Presearch per a search-query: " << nanosecondsperquery_pre << " nanoseconds.\n";
    std::cerr << "Presearch space used: " << (((float)(2 * dm.size_node_lefts) / (float)(1024 * 1024))) << " MB.\n";

    std::cerr << "Search per query: " << nanosecondsperquery << " nanoseconds.\n";
    std::cerr << "Search per query (send, search and copy the results): " << ns_per_query_total << " nanoseconds.\n";
    std::cerr << "TOTAL: (read, send, search, and copy the results): " << millis_total << " ms\n";
    std::cerr << "TOTAL: Search per query (read, send, search, and copy the results): " << ns_per_query_real_total << " nanoseconds.\n";
  }

}