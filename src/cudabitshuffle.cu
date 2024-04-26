/*
 * CUDA Kernel for bit shuffling
 */

#include "cudabitshuffle.hpp"
#include <iostream>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s: %s\n", cudaGetErrorName(err),                    \
             cudaGetErrorString(err));                                         \
    }                                                                          \
  } while (0)

//
inline auto cuda_error_string(cudaError_t err) {
  const char *err_name = cudaGetErrorName(err);
  const char *err_str = cudaGetErrorString(err);
  printf("CUDA Error: %s: %s\n", err_name, err_str);
}
/// Raise an exception IF CUDA is in an error state, with the name and
/// description
inline auto cuda_throw_error() -> void {
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s: %s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
  }
}

__global__ void cuda_bitshuffle(unsigned int *d_input, unsigned int *d_output,
                                int numElements, int numBits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    unsigned int input = d_input[idx];
    unsigned int output = 0;
    for (int i = 0; i < numBits; i++) {
      output |= ((input >> i) & 1) << (numBits - 1 - i);
    }
    d_output[idx] = output;
  }
}

__global__ void print_array_kernel(uint8_t *d_buffer, int length, int index) {
  int limit = min(index + 50, length);
  for (int i = index; i < limit; i++) {
    printf("%d ", d_buffer[i]);
  }
  printf("\n");
}

__global__ void test() { printf("Hello from CUDA\n"); }

void nvc_decompress(uint8_t *d_buffer) {
  using namespace nvcomp;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  printf("Creating manager\n");
  auto decomp_nvcomp_manager = create_manager(d_buffer, stream);

  printf("Configuring decompression\n");
  DecompressionConfig decomp_config =
      decomp_nvcomp_manager->configure_decompression(d_buffer);
  uint8_t *res_decomp_buffer;

  printf("Allocating memory\n");
  size_t available_memory, total_memory;
  CUDA_CHECK(cudaMemGetInfo(&available_memory, &total_memory));
  printf("Total memory: %lu bytes, Available memory: %lu bytes\n", total_memory,
         available_memory);

  if (decomp_config.decomp_data_size > available_memory) {
    printf("Not enough memory for decompressed data\n");
    return;
  }
  CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  size_t d_buffer_size;
  CUDA_CHECK(cudaMemGetInfo(nullptr, &d_buffer_size));
  printf("Size of d_buffer: %lu bytes\n", d_buffer_size);

  size_t res_decomp_buffer_size;
  CUDA_CHECK(cudaMemGetInfo(nullptr, &res_decomp_buffer_size));
  printf("Size of res_decomp_buffer: %lu bytes\n", res_decomp_buffer_size);

  printf("Decompressing\n");
  // TODO: Fix this
  /*

  Decompressing
  terminate called after throwing an instance of 'std::runtime_error'
    what():  Encountered Cuda Error: 2: 'out of memory'.
  Aborted (core dumped)

  */
  decomp_nvcomp_manager->decompress(res_decomp_buffer, d_buffer, decomp_config);

  printf("Printing decompressed array\n");
  print_array(res_decomp_buffer, decomp_config.decomp_data_size, 0);
}

void run_test() {
  test<<<1, 1>>>();
  cuda_throw_error();
  cudaDeviceSynchronize();
}

void print_array(uint8_t *d_buffer, int length, int index) {
  print_array_kernel<<<1, 1>>>(d_buffer, length, index);
  cuda_throw_error();
  cudaDeviceSynchronize();
}