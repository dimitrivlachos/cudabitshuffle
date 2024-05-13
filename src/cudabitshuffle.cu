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
  } else {
    printf("No CUDA error\n");
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

    if (i < 10 || i > 4400) {
      printf("Current pointer %d: %p\n", i, current_ptr);
    }

    // Set the pointer to the compressed block
    d_compressed_ptrs[i] = (void *)current_ptr;
  }
}

void block_offset_to_pointers(const uint8_t *d_compressed_data,
                              const std::vector<int> &block_offsets,
                              void **d_compressed_ptrs) {
  // Copy block_offsets to device memory
  int *d_block_offsets;
  cudaMalloc((void **)&d_block_offsets, block_offsets.size() * sizeof(int));
  cudaMemcpy(d_block_offsets, block_offsets.data(),
             block_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel
  int num_blocks = block_offsets.size();
  dim3 blocks((num_blocks + 255) /
              256); // Create enough blocks to cover all the indices
  dim3 threads(256);
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

  // The pointers on the GPU, to the compressed chunks
  void **d_compressed_ptrs;
  cudaMalloc(&d_compressed_ptrs, sizeof(uint8_t *) * batch_size);

  // copy block offsets to device
  int *device_block_offsets;
  cudaMalloc(&device_block_offsets,
             sizeof(int) * absolute_block_offsets.size());

  block_offset_to_pointers(d_compressed_data, absolute_block_offsets,
                           d_compressed_ptrs);

  // // Allocate device memory for the compressed bytes
  size_t *d_compressed_bytes;
  cudaMalloc(&d_compressed_bytes, sizeof(size_t) * batch_size);
  d_compressed_bytes = (size_t *)compressed_size;

  // // Allocate device memory for the uncompressed bytes
  size_t *d_uncompressed_bytes;
  cudaMalloc(&d_uncompressed_bytes, sizeof(size_t) * batch_size);
  d_uncompressed_bytes = (size_t *)decompressed_size;

  // // Get the size of the decompressed data
  nvcompBatchedLZ4GetDecompressSizeAsync(d_compressed_ptrs, d_compressed_bytes,
                                         d_uncompressed_bytes, batch_size,
                                         stream);

  // Allocate the temporary buffer for the decompressed data
  size_t decomp_temp_bytes;
  nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size,
                                        &decomp_temp_bytes);
  void *d_decomp_temp;
  cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

  printf("Decompressed size: %zu\n", decompressed_size);

  // Allocate statuses
  nvcompStatus_t *device_statuses;
  cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * batch_size);

  // Also allocate an array to store the actual_uncompressed_bytes.
  // Note that we could use nullptr for this. We already have the
  // actual sizes computed during the call to
  // nvcompBatchedLZ4GetDecompressSizeAsync.
  size_t *d_actual_uncompressed_bytes;
  cudaMalloc(&d_actual_uncompressed_bytes, sizeof(size_t) * batch_size);

  void **d_uncompressed_ptrs;
  cudaMalloc(&d_uncompressed_ptrs, sizeof(uint8_t *) * batch_size);

  // Decompress the data
  // This decompresses each input, device_compressed_ptrs[i], and places the
  // decompressed result in the corresponding output list,
  // device_uncompressed_ptrs[i]. It also writes the size of the uncompressed
  // data to d_uncompressed_bytes[i].
  nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
      d_compressed_ptrs, d_compressed_bytes, d_uncompressed_bytes,
      d_actual_uncompressed_bytes, batch_size, d_decomp_temp, decomp_temp_bytes,
      d_uncompressed_ptrs,
      device_statuses, // make nullptr to improve throughput
      stream);

  if (decomp_res != nvcompSuccess) {
    printf("Error in decompression\n");
  } else {
    printf("Decompression successful\n");
  }

  // Wait for the decompression to finish
  cudaStreamSynchronize(stream);
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