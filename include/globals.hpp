#pragma once
#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include "../include/utils/helpers.hpp"
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

using u64 = uint64_t;
using u32 = uint32_t;
using u8 = uint8_t;
using int64 = int64_t;

typedef u8 byte32_t __attribute__((vector_size(32 * sizeof(u8))));

#define REGULAR_POPPY true // true: regular poppy with absolute basic block counts,
// false = poppy with cumulative basic block counts

// This must be changed with other than nvidia a100
#define DEVICE_IS_NVIDIA_A100 true

// #define DEBUG
#define BENCHMARK

#ifdef DEBUG
#define DEBUG_CODE(expr) expr
#else
#define DEBUG_CODE(expr) \
  {                      \
  }
#endif

#ifdef BENCHMARK
#define BENCHMARK_CODE(expr) expr
#else
#define BENCHMARK_CODE(expr) \
  {                          \
  }
#endif

#define START_TIME std::chrono::high_resolution_clock::now()
#define STOP_TIME std::chrono::high_resolution_clock::now()
// #define PRINT_DURATION_IN_MILLISECONDS(text, start, stop) (std::err << text << (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) << " milliseconds.\n");
#define DURATION_IN_MILLISECONDS(start, stop) ((float)(((double)(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count())) / 1000.0))

#ifdef DEBUG
#define DEBUG_BEFORE_DESTRUCT(msg)        \
  fprintf(stderr, "Before destructing "); \
  fprintf(stderr, msg);                   \
  fprintf(stderr, "\n");
#else
#define DEBUG_BEFORE_DESTRUCT(msg) \
  {                                \
  }
#endif

#ifdef DEBUG
#define DEBUG_AFTER_DESTRUCT(msg)        \
  fprintf(stderr, "After destructing "); \
  fprintf(stderr, msg);                  \
  fprintf(stderr, "\n");
#else
#define DEBUG_AFTER_DESTRUCT(msg) \
  {                               \
  }
#endif