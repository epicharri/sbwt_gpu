//#include "../include/search.hpp"
#include "../include/globals.hpp"
#include "../include/gpu/device_memory.hpp"
#include "../include/matrix_sbwt.hpp"
#include "../include/parameters.hpp"

#include "../include/epicseq/kmer_search.hpp"

int main(int argc, char **argv)
{

  cudaDeviceProp prop;
  epic::Parameters parameters;
  if (parameters.read_arguments(argc, argv, prop))
    return 0;
  epic::gpu::DeviceMemory dm;
  epic::MatrixSBWT sbwt;
  if (sbwt.create(dm, parameters, DEVICE_IS_NVIDIA_A100))
    return 0;
  fprintf(stderr, "%f ms for reading the bit vector files, constructing the rank data structures, and sending to the device.\n", sbwt.millis_before_presearch);

  epic::KmerSearch searcher(parameters.fileQueries, parameters);
  if (searcher.search(dm))
    return 0;
  return 0;
}
