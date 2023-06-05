#pragma once
#include "../epicseq/kmer_data.hpp"
#include "../globals.hpp"
#include "../gpu/device_memory.hpp"
#include "../gpu/device_stream.hpp"
#include "../parameters.hpp"
#include <charconv>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <vector>

namespace epic
{

  struct ResultsPersister
  {
    u64 number_of_errors = 0ULL;
    u64 number_of_minus_ones = 0ULL;
    float millis_copy_results = 0.0;
    std::vector<std::string> read_answer_lines(std::string);
    int save_results_to_file(std::vector<int64> &, epic::KmerData &, epic::gpu::DeviceMemory &, epic::Parameters &, u32);
    int64 max_space_needed_for_one_number(std::vector<int64> &, epic::KmerData &, epic::gpu::DeviceMemory &);
    int64 max_space_needed_for_results(std::vector<int64> &, epic::KmerData &, epic::gpu::DeviceMemory &);
    ResultsPersister() = default;
  };

  int64 ResultsPersister::max_space_needed_for_one_number(std::vector<int64> &start_positions, epic::KmerData &kmer_data, epic::gpu::DeviceMemory &dm)
  {
    int64 largest_number_to_be_printed = dm.size_A * 8;
    int max_number_length = 0;
    // Since the size of a bit vector is greater than 0, the following works correctly.
    while (largest_number_to_be_printed > 0)
    {
      max_number_length += 1;
      largest_number_to_be_printed /= 10;
    }
    return (int64)(max_number_length + 1) + 1; // White space and new line.
  }

  int64 ResultsPersister::max_space_needed_for_results(std::vector<int64> &start_positions, epic::KmerData &kmer_data, epic::gpu::DeviceMemory &dm)
  {
    int64 largest_number_to_be_printed = dm.size_A * 8;
    int max_number_length = 0;
    // Since the size of a bit vector is greater than 0, the following works correctly.
    while (largest_number_to_be_printed > 0)
    {
      max_number_length += 1;
      largest_number_to_be_printed /= 10;
    }
    return (int64)kmer_data.number_of_positions * (max_number_length + 1) + start_positions.size();
  }

  int ResultsPersister::save_results_to_file(std::vector<int64> &start_positions, epic::KmerData &kmer_data, epic::gpu::DeviceMemory &dm, epic::Parameters &parameters, u32 kmer_size)
  {
    CHECK(cudaDeviceSynchronize());
    dm.device_stream.start_timer();

    CHECK(cudaMemcpy(
        kmer_data.positions_and_results,
        kmer_data.d_positions_and_results,
        kmer_data.size_positions_and_results,
        cudaMemcpyDeviceToHost));
    dm.device_stream.stop_timer();
    millis_copy_results = dm.device_stream.duration_in_millis();
    fprintf(stderr, "Transfer the results from the device to the host memory: %f ms\n", millis_copy_results);

    if (parameters.store_results == false)
    {
      fprintf(stderr, "The results are not stored to the file.\n");
      return 0;
    }

    int64 i_results = 0ULL;
    u64 row_length = 0ULL;

    u32 size_of_buffer = max_space_needed_for_one_number(start_positions, kmer_data, dm);
    char buf[size_of_buffer];

    for (int64 i = 0; i < (start_positions.size() - 1); i++)
    {
      u64 i_row = 0;
      row_length = start_positions[i + 1] - start_positions[i] - kmer_size + 1;
      for (; i_row < row_length; i_row++)
      {
        const std::to_chars_result number_as_chars = std::to_chars(
            buf,
            buf + size_of_buffer,
            (int64)kmer_data.positions_and_results[i_row + i_results]);
        fprintf(stdout, "%.*s ", static_cast<int>(number_as_chars.ptr - buf), buf);
      }
      fprintf(stdout, "\n");
      i_results += row_length;
    }

    fprintf(stderr, "The results stored successfully to a file.\n");
    return 0;
  }

  /*
    int ResultsPersister::save_results_to_file(std::vector<int64> &start_positions, epic::KmerData &kmer_data, epic::gpu::DeviceMemory &dm, epic::Parameters &parameters, u32 kmer_size)
    {
      CHECK(cudaDeviceSynchronize());
      dm.device_stream.start_timer();
      CHECK(cudaMemcpy(
          kmer_data.positions_and_results,
          kmer_data.d_positions_and_results,
          kmer_data.size_positions_and_results,
          cudaMemcpyDeviceToHost));
      dm.device_stream.stop_timer();
      millis_copy_results = dm.device_stream.duration_in_millis();
      fprintf(stderr, "Transfer the results from the device to the host memory: %f ms\n", millis_copy_results);

      if (parameters.store_results == false)
      {
        fprintf(stderr, "The results are not stored to the file.\n");
        return 0;
      }

      int64 i_results = 0ULL;
      u64 row_length = 0ULL;

      u32 size_of_buffer = max_space_needed_for_one_number(start_positions, kmer_data, dm);
      char buf[size_of_buffer];

      for (int64 i = 0; i < (start_positions.size() - 1); i++)
      {
        u64 i_row = 0;
        row_length = start_positions[i + 1] - start_positions[i] - kmer_size + 1;
        for (; i_row < row_length; i_row++)
        {
          const std::to_chars_result number_as_chars = std::to_chars(
              buf,
              buf + size_of_buffer,
              (int64)kmer_data.positions_and_results[i_row + i_results]);
          fprintf(stdout, "%.*s ", static_cast<int>(number_as_chars.ptr - buf), buf);
        }
        fprintf(stdout, "\n");
        i_results += row_length;
      }

      fprintf(stderr, "The results stored successfully to a file.\n");
      return 0;
    }
  */
}