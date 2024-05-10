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
 * @brief: Get the pointers to the blocks in the buffer
 * @param d_buffer: Pointer to the buffer
 * @param d_block_pointers: Pointer to the array of pointers to the blocks
 */
// __global__ void getBlockPointersKernel(uint8_t *d_buffer,
//                                        uint8_t **d_block_pointers,
//                                        uint32_t image_size) {
//   printf("Getting block pointers K\n");
//   uint8_t *block = d_buffer + 12;
//   // uint32_t image_size = (uint32_t) * (uint64_t *)d_buffer;
//   uint32_t n_block = image_size / 8192;
//   if (image_size % 8192)
//     n_block++;

//   printf("Image size: %d\n", image_size);
//   printf("Number of blocks: %d\n", n_block);

//   for (int i = 0; i < n_block; i++) {
//     d_block_pointers[i] = block;
//     printf("Block %d: %p\n", i, block);
//     uint32_t header = byteswap32(block);
//     printf("Header: %p\n", header);
//     uint32_t next = *(uint32_t *)(header);
//     printf("Next: %d\n", next);
//     block += next + 4;
//   }
//   printf("Done getting block pointers K\n");
// }

// /**
//  * @brief: Get the size of the decompressed data
//  * @param d_buffer: Pointer to the buffer
//  * @param d_block_pointers: Pointer to the array of pointers to the blocks
//  */
// void getBlockPointers(uint8_t *d_buffer, uint8_t **d_block_pointers,
//                       uint32_t image_size) {
//   fmt::print("Getting block pointers\n");
//   getBlockPointersKernel<<<1, 1>>>(d_buffer, d_block_pointers, image_size);
//   cuda_throw_error();
//   cudaDeviceSynchronize();
//   fmt::print("Done getting block pointers\n");
// }

void decompress_lz4_gpu(const uint8_t *compressed_data, size_t compressed_size,
                        uint8_t *decompressed_data, size_t decompressed_size,
                        const std::vector<int> &block_offsets) {
  fmt::print(fmt::fg(fmt::color::green), "Decompressing LZ4 data on GPU\n");
  // Create a CUDA stream
  // cudaStream_t stream;
  // cudaStreamCreate(&stream);

  // Print block offsets
  for (int i = 0; i < block_offsets.size(); i++) {
    fmt::print("Block offset: {}\n", block_offsets[i]);
  }

  // // Set chunk size and batch size
  // size_t chunk_size = 8192;
  // size_t batch_size = (compressed_size + chunk_size - 1) / chunk_size;

  // // Allocate device memory for list of pointers to compressed blocks
  // void **device_compressed_ptrs;
  // cudaMalloc(&device_compressed_ptrs, sizeof(uint8_t *) * batch_size);

  // // Get the pointers to the compressed blocks
  // getBlockPointers((uint8_t *)compressed_data,
  //                  (uint8_t **)device_compressed_ptrs, compressed_size);

  // // Allocate device memory for the compressed bytes
  // size_t *device_compressed_bytes;
  // cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);
  // device_compressed_bytes = (size_t *)compressed_size;

  // // Allocate device memory for the uncompressed bytes
  // size_t *device_uncompressed_bytes;
  // cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size);
  // device_uncompressed_bytes = (size_t *)decompressed_size;

  // // Get the size of the decompressed data
  // nvcompBatchedLZ4GetDecompressSizeAsync(
  //     device_compressed_ptrs, device_compressed_bytes,
  //     device_uncompressed_bytes, batch_size, stream);

  // fmt::print("Decompressed size: {}\n", device_uncompressed_bytes[0]);
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