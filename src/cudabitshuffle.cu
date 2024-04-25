/*
 * CUDA Kernel for bit shuffling
 */

#include "cudabitshuffle.hpp"
#include <iostream>
#include <stdio.h>

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