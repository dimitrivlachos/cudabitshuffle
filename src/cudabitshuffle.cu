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

__global__ void
block_offset_to_pointers_kernel(const uint8_t *d_compressed_data,
                                const int *d_block_offsets,
                                void **d_compressed_ptrs, int num_blocks) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < num_blocks) {
    if (i > 4400) {
      printf("Index: %d\n", i);
    }
    if (i == 4419) {
      printf("Too high");
    }
    // Initialize the pointer to the beginning of the compressed data
    const uint8_t *current_ptr = d_compressed_data;

    // Move the pointer to the block offset
    current_ptr += d_block_offsets[i];

    // Set the pointer to the compressed block
    d_compressed_ptrs[i] = (void *)current_ptr;
  }
}

void block_offset_to_pointers(const uint8_t *d_compressed_data,
                              const std::vector<int> &block_offsets,
                              void **d_compressed_ptrs) {
  printf("Converting block offsets to pointers\n");
  // Copy block_offsets to device memory
  int *d_block_offsets;
  cudaMalloc((void **)&d_block_offsets, block_offsets.size() * sizeof(int));
  cudaMemcpy(d_block_offsets, block_offsets.data(),
             block_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel
  int num_blocks = block_offsets.size();
  printf("Launching kernel - num_blocks: %d\n", num_blocks);
  printf("Size of d_block_offsets: %d\n", block_offsets.size());
  dim3 blocks((num_blocks + 255) /
              256); // Create enough blocks to cover all the indices
  dim3 threads(256);
  printf("Blocks: %d, Threads: %d\n", blocks.x, threads.x);
  printf("block_offsets size: %d\n", block_offsets.size());
  printf("block_offset 4418: %d\n", block_offsets[4418]);
  printf("block_offset 4419: %d\n", block_offsets[4419]);
  block_offset_to_pointers_kernel<<<blocks, threads>>>(
      d_compressed_data, d_block_offsets, d_compressed_ptrs, num_blocks);

  cuda_throw_error();
  cudaDeviceSynchronize();

  // Free device memory
  cudaFree(d_block_offsets);
}

void decompress_lz4_gpu(const uint8_t *d_compressed_data,
                        size_t compressed_size, uint8_t *d_decompressed_data,
                        size_t decompressed_size,
                        const std::vector<int> &absolute_block_offsets) {
  fmt::print(fmt::fg(fmt::color::green), "Decompressing LZ4 data on GPU\n");
  // Create a CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // // Set chunk size and batch size
  size_t chunk_size = 8192;
  size_t batch_size = ((compressed_size + chunk_size - 1) / chunk_size) + 1;

  // // Allocate device memory for list of pointers to compressed blocks
  void **device_compressed_ptrs;
  cudaMalloc(&device_compressed_ptrs, sizeof(uint8_t *) * batch_size);
  printf("Batch size: %d\n", batch_size);

  // copy block offsets to device
  int *device_block_offsets;
  cudaMalloc(&device_block_offsets,
             sizeof(int) * absolute_block_offsets.size());

  block_offset_to_pointers(d_compressed_data, absolute_block_offsets,
                           device_compressed_ptrs);

  // Print the first 10 block pointers
  for (int i = 0; i < 10; i++) {
    fmt::print("Block pointer: {}\n", device_compressed_ptrs[i]);
  }

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