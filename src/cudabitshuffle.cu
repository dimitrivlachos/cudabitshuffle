/*
 * CUDA Kernel for bit shuffling
 */

#include "cudabitshuffle.hpp"

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

// void nvc_compress(uint8_t *d_buffer) {
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);

//   // Inirialize data on host
//   size_t *host_uncompressed_bytes;
//   const size_t chunk_size = 8192;
//   const batch_size =
// }

/**
 * @brief: Return the byteswapped value of a 64-bit header
 * @param ptr: Pointer to the 64-bit header
 * @return: The byteswapped value of the header
 */
__device__ uint64_t byteswap64(const void *ptr) {
  uint64_t value;
  memcpy(&value, ptr, sizeof(uint64_t));
  uint8_t *bytes = (uint8_t *)&value;
  uint8_t tmp;
  tmp = bytes[0];
  bytes[0] = bytes[7];
  bytes[7] = tmp;
  tmp = bytes[1];
  bytes[1] = bytes[6];
  bytes[6] = tmp;
  tmp = bytes[2];
  bytes[2] = bytes[5];
  bytes[5] = tmp;
  tmp = bytes[3];
  bytes[3] = bytes[4];
  bytes[4] = tmp;
  memcpy(&value, bytes, sizeof(uint64_t));
  return value;
}

/**
 * @brief: Return the byteswapped value of a 32-bit header
 * @param ptr: Pointer to the 32-bit header
 * @return: The byteswapped value of the header
 */
__device__ uint32_t byteswap32(void *ptr) {
  uint32_t value;
  memcpy(&value, ptr, sizeof(uint32_t));
  uint8_t *bytes = (uint8_t *)&value;
  uint8_t tmp;
  tmp = bytes[0];
  bytes[0] = bytes[3];
  bytes[3] = tmp;
  tmp = bytes[1];
  bytes[1] = bytes[2];
  bytes[2] = tmp;
  memcpy(&value, bytes, sizeof(uint32_t));
  return value;
}

/**
 * @brief: Get the pointers to the blocks in the buffer
 * @param d_buffer: Pointer to the buffer
 * @param d_block_pointers: Pointer to the array of pointers to the blocks
 */
__device__ void getBlockPointers(uint8_t *d_buffer,
                                 uint8_t **d_block_pointers) {
  uint8_t *block = d_buffer + 12;
  uint32_t image_size = (uint32_t) * (uint64_t *)d_buffer;
  uint32_t n_block = image_size / 8192;
  if (image_size % 8192)
    n_block++;
  for (int i = 0; i < n_block; i++) {
    d_block_pointers[i] = block;
    byteswap32(block);
    uint32_t next = *(uint32_t *)block;
    block += next + 4;
  }
}

void decompress_lz4_gpu(const uint8_t *compressed_data, size_t compressed_size,
                        uint8_t *decompressed_data, size_t decompressed_size) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t chunk_size = 8192;
  size_t batch_size = (compressed_size + chunk_size - 1) / chunk_size;
  void **device_compressed_ptrs;
  size_t *device_compressed_bytes;
  void **device_uncompressed_bytes;

  nvcompBatchedLZ4GetDecompressSizeAsync(
      device_compressed_ptrs, device_compressed_bytes,
      device_uncompressed_bytes, batch_size, stream);
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