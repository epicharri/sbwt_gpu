#pragma once
#include "../bit_vector.hpp"
#include "../enums.hpp"
#include "../epicseq/kmer_data.hpp"
#include "../globals.hpp"
#include "../gpu/device_memory.hpp"
#include "../gpu/device_stream.hpp"

#include "../gpu/search_and_presearch_launchers.hpp"
#include "../parameters.hpp"
#include <fstream>
#include <iostream>

namespace epic
{

  struct SearchCaller
  {
    float millis = 0.0;
    int search(epic::Parameters &, epic::gpu::DeviceMemory &, epic::KmerData &);
    int search_in_gpu(epic::Parameters &, epic::gpu::DeviceMemory &, epic::KmerData &);
    int search_using_gpu_memory_for_queries(
        epic::Parameters &,
        epic::gpu::DeviceMemory &,
        epic::KmerData &);
    //    void print_benchmark_information(epic::KmerString &, epic::Parameters &, epic::gpu::DeviceMemory &);
    SearchCaller() = default;
  };

  int SearchCaller::search_using_gpu_memory_for_queries(
      epic::Parameters &parameters,
      epic::gpu::DeviceMemory &dm,
      epic::KmerData &kmer_data)
  {

    if (search_in_gpu(parameters, dm, kmer_data))
    {
      fprintf(stderr, "Search in GPU did not succeed.\n");
      return 1;
    }
    /*
    cudaDeviceSynchronize();
    dm.device_stream.start_timer();
    cudaMemcpy(kmer_data.positions_and_results, kmer_data.d_positions_and_results, kmer_data.size_positions_and_results, cudaMemcpyDeviceToHost);
    dm.device_stream.stop_timer();
    millis_copy_results = dm.device_stream.duration_in_millis();

    fprintf(stderr, "SearchCaller: kmer_data.size_positions_and_results = %" PRIu64 "\n", kmer_data.size_positions_and_results);

    for (int i = 0; i < 1000; i++)
    {
      fprintf(stderr, "%" PRIu64 " ", (kmer_data.positions_and_results[i]));
    }
    */

    return 0;
  }

  int SearchCaller::search(
      epic::Parameters &parameters,
      epic::gpu::DeviceMemory &dm,
      epic::KmerData &kmer_data)
  {

    if (search_using_gpu_memory_for_queries(parameters, dm, kmer_data))
    {
      fprintf(stderr, "Reading the queries and searching did not succeed.\n");
      return 1;
    }

    // This is to test if queries are correct. In production there must be saving the results to a file.
    /*
    epic::CompareToAnswers tester;
    tester.check(kmer_string, parameters.fileAnswers);
    print_benchmark_information(kmer_string, parameters, dm);
    */
    // Rewrite the tester and its call.
    return 0;
  }

  int SearchCaller::search_in_gpu(epic::Parameters &parameters, epic::gpu::DeviceMemory &dm, epic::KmerData &kmer_data)
  {

    u64 block_size, grid_size;
    block_size = parameters.threads_per_block;
    grid_size = (kmer_data.number_of_positions_padded + block_size - 1ULL) / block_size;

    millis = 0.0;
    int rank_version = parameters.rank_structure_version;

    if (DEVICE_IS_NVIDIA_A100)
    {
      CHECK(cudaDeviceSetLimit(
          cudaLimitMaxL2FetchGranularity,
          parameters.device_set_limit_search));
    }
    //    cudaDeviceSynchronize();
    if (parameters.with_presearch)
    {
      switch (parameters.bits_in_superblock)
      {
      case 256:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<256, false, epic::kind::poppy, true>(grid_size, block_size, dm, kmer_data);
        else if (rank_version == epic::kind::cum_poppy)
          millis = epic::gpu::launch_search_kernel<256, false, epic::kind::cum_poppy, true>(grid_size, block_size, dm, kmer_data);
        break;
      case 512:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<512, false, epic::kind::poppy, true>(grid_size, block_size, dm, kmer_data);
        else if (rank_version == epic::kind::cum_poppy)
          millis = epic::gpu::launch_search_kernel<512, false, epic::kind::cum_poppy, true>(grid_size, block_size, dm, kmer_data);
        break;
      case 1024:
        if (parameters.with_shuffles)
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<1024, true, epic::kind::poppy, true>(grid_size, block_size, dm, kmer_data);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<1024, true, epic::kind::cum_poppy, true>(grid_size, block_size, dm, kmer_data);
        }
        else
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<1024, false, epic::kind::poppy, true>(grid_size, block_size, dm, kmer_data);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<1024, false, epic::kind::cum_poppy, true>(grid_size, block_size, dm, kmer_data);
        }
        break;
      case 2048:
        if (parameters.with_shuffles)
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<2048, true, epic::kind::poppy, true>(grid_size, block_size, dm, kmer_data);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<2048, true, epic::kind::cum_poppy, true>(grid_size, block_size, dm, kmer_data);
        }
        else
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<2048, false, epic::kind::poppy, true>(grid_size, block_size, dm, kmer_data);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<2048, false, epic::kind::cum_poppy, true>(grid_size, block_size, dm, kmer_data);
        }
        break;
      case 4096:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<4096, false, epic::kind::poppy, true>(grid_size, block_size, dm, kmer_data);
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
          millis = epic::gpu::launch_search_kernel<256, false, epic::kind::poppy, false>(grid_size, block_size, dm, kmer_data);
        else if (rank_version == epic::kind::cum_poppy)
          millis = epic::gpu::launch_search_kernel<256, false, epic::kind::cum_poppy, false>(grid_size, block_size, dm, kmer_data);
        break;
      case 512:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<512, false, epic::kind::poppy, false>(grid_size, block_size, dm, kmer_data);
        else if (rank_version == epic::kind::cum_poppy)
          millis = epic::gpu::launch_search_kernel<512, false, epic::kind::cum_poppy, false>(grid_size, block_size, dm, kmer_data);
        break;
      case 1024:
        if (parameters.with_shuffles)
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<1024, true, epic::kind::poppy, false>(grid_size, block_size, dm, kmer_data);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<1024, true, epic::kind::cum_poppy, false>(grid_size, block_size, dm, kmer_data);
        }
        else
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<1024, false, epic::kind::poppy, false>(grid_size, block_size, dm, kmer_data);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<1024, false, epic::kind::cum_poppy, false>(grid_size, block_size, dm, kmer_data);
        }
        break;
      case 2048:
        if (parameters.with_shuffles)
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<2048, true, epic::kind::poppy, false>(grid_size, block_size, dm, kmer_data);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<2048, true, epic::kind::cum_poppy, false>(grid_size, block_size, dm, kmer_data);
        }
        else
        {
          if (rank_version == epic::kind::poppy)
            millis = epic::gpu::launch_search_kernel<2048, false, epic::kind::poppy, false>(grid_size, block_size, dm, kmer_data);
          else if (rank_version == epic::kind::cum_poppy)
            millis = epic::gpu::launch_search_kernel<2048, false, epic::kind::cum_poppy, false>(grid_size, block_size, dm, kmer_data);
        }
        break;
      case 4096:
        if (rank_version == epic::kind::poppy)
          millis = epic::gpu::launch_search_kernel<4096, false, epic::kind::poppy, false>(grid_size, block_size, dm, kmer_data);
        // Only poppy is supported with the superblock size 4096.
        break;
      }
    }
    return 0;
  }

  /*
    void SearchCaller::print_benchmark_information(epic::KmerString &kmer_string, epic::Parameters &parameters, epic::gpu::DeviceMemory &dm)
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
  */
}