#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <functional>

#define CHECK(x)               \
  if (epic::gpu::check(x, #x)) \
  {                            \
    return 1;                  \
  }

#define CHECK_WITHOUT_RETURN(x) (epic::gpu::check_without_print_and_return(x, #x))

#define deviceMemcpyHostToDevice cudaMemcpyHostToDevice
#define deviceMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define deviceError_t cudaError_t
#define deviceStream_t cudaStream_t
#define deviceEvent_t cudaEvent_t
#define deviceDeviceProp cudaDeviceProp
#define deviceEventElapsedTime cudaEventElapsedTime
#define deviceEventSynchronize cudaEventSynchronize
#define deviceEventCreate cudaEventCreate
#define deviceEventDestroy cudaEventDestroy
#define deviceEventRecord cudaEventRecord
#define deviceStreamCreate cudaStreamCreate
#define deviceStreamDestroy cudaStreamDestroy
#define deviceStreamSynchronize cudaStreamSynchronize
#define deviceMalloc cudaMalloc
#define deviceMallocAsync cudaMallocAsync
#define deviceFreeAsync cudaFreeAsync
#define deviceFree cudaFree
#define deviceMemGetInfo cudaMemGetInfo
#define deviceMallocManaged cudaMallocManaged
#define deviceMemPrefetchAsync cudaMemPrefetchAsync
#define deviceMemcpyAsync cudaMemcpyAsync
#define deviceMemcpy cudaMemcpy
#define deviceGetDeviceProperties cudaGetDeviceProperties
#define deviceDeviceSynchronize cudaDeviceSynchronize
#define deviceGetErrorString cudaGetErrorString
#define deviceGetLastError cudaGetLastError
#define deviceDeviceSetLimit cudaDeviceSetLimit                         // Maybe only in NVIDIA A100
#define deviceLimitMaxL2FetchGranularity cudaLimitMaxL2FetchGranularity // Maybe only in NVIDIA A100

namespace epic
{
  namespace gpu

  {
    // Helper functions:
    inline void get_and_print_last_error()
    {
      cudaError_t err = cudaGetLastError();
      fprintf(stderr, "Last CUDA error code %d (%s).\n", err, cudaGetErrorString(err));
    }

    inline void print_device_error(deviceError_t err, const char *msg)
    {
      fprintf(stderr, "CUDA error %d (%s): (%s)\n", err, cudaGetErrorString(err), msg);
    }

    inline void print_device_error_if_any(deviceError_t err, const char *msg)
    {
      if (err)
        fprintf(stderr, "CUDA error %d (%s): (%s)\n", err, cudaGetErrorString(err), msg);
    }
    inline int check(deviceError_t err, const char *context)
    {
      if (err != cudaSuccess)
      {
        fprintf(stderr, "CUDA error %d (%s) happened here: %s\n", err, cudaGetErrorString(err), context);
        return 1;
      }
      return 0;
    }

    inline int check_without_print_and_return(deviceError_t err, const char *context)
    {
      if (err != cudaSuccess)
      {
        return 1;
      }
      return 0;
    }
  }
}