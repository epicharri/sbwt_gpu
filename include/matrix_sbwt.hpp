#pragma once
#include "../include/bit_vector.hpp"
#include "../include/enums.hpp"
#include "../include/globals.hpp"
#include "../include/gpu/device_memory.hpp"
#include "../include/presearch.hpp"
#include "../include/rank_data_structures.hpp"
#include <omp.h>

namespace epic
{

  struct MatrixSBWT
  {
    u64 number_of_bits = 0ULL; // Number of bits in the bit vector.
    bool success = false;
    float millis_pre = 0.0;
    float millis_before_presearch = 0.0;
    int rank_version = epic::kind::poppy;
    int create(epic::gpu::DeviceMemory &,
               epic::Parameters &,
               bool);
    MatrixSBWT() = default;
    ~MatrixSBWT();
  };

  MatrixSBWT::~MatrixSBWT()
  {
    DEBUG_CODE(fprintf(stderr, "MatrixSBWT deleted.\n");)
  }

  int MatrixSBWT::create(
      epic::gpu::DeviceMemory &dm,
      epic::Parameters &parameters,
      bool device_is_nvidia_a100)
  {
    auto start_create_sbwt_without_reading_files = START_TIME;
    auto start_creating_sbwt = START_TIME;
    dm.device_stream.start_timer();
    dm.set_lengths_of_kmer_and_presearch_mer(parameters.k, parameters.k_presearch);
    int rank_version = parameters.rank_structure_version;
    epic::PreSearch pre_searcher(parameters.k_presearch);
    dm.set_sizes_of_node_lefts_and_rights(pre_searcher.number_of_words);
    epic::BitVector A, C, G, T;
    epic::RankDataStructures rank_A, rank_C, rank_G, rank_T;
    if (A.create(parameters.filename_A) |
        C.create(parameters.filename_C) |
        G.create(parameters.filename_G) |
        T.create(parameters.filename_T))
      return 1;
    number_of_bits = A.number_of_bits;
    if ((C.number_of_bits != A.number_of_bits) |
        (G.number_of_bits != A.number_of_bits) |
        (T.number_of_bits != A.number_of_bits))
      return 1;
    rank_A.create(parameters.bits_in_superblock, rank_version, A);
    rank_C.create(parameters.bits_in_superblock, rank_version, C);
    rank_G.create(parameters.bits_in_superblock, rank_version, G);
    rank_T.create(parameters.bits_in_superblock, rank_version, T);
    dm.set_sizes_of_rank_data_structures(
        rank_A.number_of_words_padded_layer_0,
        rank_A.number_of_words_padded_layer_12);
    dm.set_sizes_of_bit_vectors(A.number_of_words_padded);
    if (dm.allocate_device_memory())
      return 1;
    int error_A, error_C, error_G, error_T;
    error_A = error_C = error_G = error_T = 0;
    auto stop_create_sbwt_without_reading_files = STOP_TIME;
    auto start_to_read_sbwt_files = START_TIME;
#pragma omp parallel
    {
#pragma omp single
      {
#pragma omp task private(error_A)
        {
          error_A = A.read();
          if (!error_A)
          {
            error_A = rank_A.construct(A, 0);
          }
        }

#pragma omp task private(error_C)
        {
          error_C = C.read();
          if (!error_C)
          {
            error_C = rank_C.construct(C, 0);
          }
        }

#pragma omp task private(error_G)
        {
          error_G = G.read();
          if (!error_G)
          {
            error_G = rank_G.construct(G, 0);
          }
        }

#pragma omp task private(error_T)
        {
          error_T = T.read();
          if (!error_T)
          {
            error_T = rank_T.construct(T, 0);
          }
        }
      }
    }
    auto stop_to_read_sbwt_files = STOP_TIME;

    auto start_create_sbwt_without_reading_files_after_reading = START_TIME;

    if (!error_A)
    {
      error_A = CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.A, A.data, dm.size_A, cudaMemcpyHostToDevice, dm.device_stream.stream)) |
                CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.L0_A, rank_A.layer_0, dm.size_L0_A, cudaMemcpyHostToDevice, dm.device_stream.stream)) |
                CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.L12_A, rank_A.layer_12, dm.size_L12_A, cudaMemcpyHostToDevice, dm.device_stream.stream));
    }
    if (!error_C)
    {

      error_C = CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.C, C.data, dm.size_C, cudaMemcpyHostToDevice, dm.device_stream.stream)) |
                CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.L0_C, rank_C.layer_0, dm.size_L0_C, cudaMemcpyHostToDevice, dm.device_stream.stream)) |
                CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.L12_C, rank_C.layer_12, dm.size_L12_C, cudaMemcpyHostToDevice, dm.device_stream.stream));
    }
    if (!error_G)
    {
      error_G = CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.G, G.data, dm.size_G, cudaMemcpyHostToDevice, dm.device_stream.stream)) |
                CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.L0_G, rank_G.layer_0, dm.size_L0_G, cudaMemcpyHostToDevice, dm.device_stream.stream)) |
                CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.L12_G, rank_G.layer_12, dm.size_L12_G, cudaMemcpyHostToDevice, dm.device_stream.stream));
    }
    if (!error_T)
    {
      error_T = CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.T, T.data, dm.size_T, cudaMemcpyHostToDevice, dm.device_stream.stream)) |
                CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.L0_T, rank_T.layer_0, dm.size_L0_T, cudaMemcpyHostToDevice, dm.device_stream.stream)) |
                CHECK_WITHOUT_RETURN(cudaMemcpyAsync(dm.L12_T, rank_T.layer_12, dm.size_L12_T, cudaMemcpyHostToDevice, dm.device_stream.stream));
    }

    if (error_A)
      return 1;
    if (error_C)
      return 1;
    if (error_G)
      return 1;
    if (error_T)
      return 1;

    u64 counts_before[5];
    counts_before[0] = 1ULL;
    counts_before[1] = counts_before[0] + rank_A.absolute_number_of_ones;
    counts_before[2] = counts_before[1] + rank_C.absolute_number_of_ones;
    counts_before[3] = counts_before[2] + rank_G.absolute_number_of_ones;
    counts_before[4] = counts_before[3] + rank_T.absolute_number_of_ones;

    CHECK(cudaMemcpyAsync(dm.counts_before, &counts_before[0], dm.size_counts_before, cudaMemcpyHostToDevice, dm.device_stream.stream));

    u64 *device_SBWT[4];
    u64 *device_L0[4];
    u64 *device_L12[4];
    device_SBWT[0] = dm.A;
    device_SBWT[1] = dm.C;
    device_SBWT[2] = dm.G;
    device_SBWT[3] = dm.T;
    device_L0[0] = dm.L0_A;
    device_L0[1] = dm.L0_C;
    device_L0[2] = dm.L0_G;
    device_L0[3] = dm.L0_T;
    device_L12[0] = dm.L12_A;
    device_L12[1] = dm.L12_C;
    device_L12[2] = dm.L12_G;
    device_L12[3] = dm.L12_T;

    CHECK(cudaMemcpyAsync(dm.SBWT, &device_SBWT[0], dm.size_SBWT, cudaMemcpyHostToDevice, dm.device_stream.stream));
    CHECK(cudaMemcpyAsync(dm.L0, &device_L0[0], dm.size_L0, cudaMemcpyHostToDevice, dm.device_stream.stream));
    CHECK(cudaMemcpyAsync(dm.L12, &device_L12[0], dm.size_L12, cudaMemcpyHostToDevice, dm.device_stream.stream));

    dm.device_stream.stop_timer();
    millis_before_presearch = dm.device_stream.duration_in_millis();
    cudaDeviceSynchronize();
    auto stop_creating_sbwt = STOP_TIME;
    float millis_stop_creating_sbwt = DURATION_IN_MILLISECONDS(start_creating_sbwt, stop_creating_sbwt);
    fprintf(stderr, "Creating the SBWT structures from memory allocation to the point before presearch takes %f ms.\n", millis_stop_creating_sbwt);
    //    cudaStreamSynchronize(dm.device_stream.stream);
    pre_searcher.call_presearch(parameters, dm, device_is_nvidia_a100); // This is blocking, but can be made unblocking by not synchronizing in DeviceStream.
    millis_pre = pre_searcher.millis_pre;
    fprintf(stderr, "Presearch takes %f ms.\n", millis_pre);

    auto stop_create_sbwt_without_reading_files_after_reading = STOP_TIME;
    float millis_before_reading_files = DURATION_IN_MILLISECONDS(start_create_sbwt_without_reading_files, stop_create_sbwt_without_reading_files);
    float millis_after_reading_files = DURATION_IN_MILLISECONDS(start_create_sbwt_without_reading_files_after_reading, stop_create_sbwt_without_reading_files_after_reading);
    float millis_to_read_files = DURATION_IN_MILLISECONDS(start_to_read_sbwt_files, stop_to_read_sbwt_files);
    fprintf(stderr, "SBWT: Read files takes %f ms.\n", millis_to_read_files);
    fprintf(stderr, "SBWT: Necessary memory allocations before reading SBWT files and constructing the rank structures takes %f ms.\n", millis_before_reading_files);
    fprintf(stderr, "SBWT: Memory transfer from host to device for the SBWT and rank indexes and presearching takes %f ms.\n", millis_after_reading_files);
    fprintf(stderr, "Presearch takes %f ms.\n", millis_pre);

    return 0;
  }
}