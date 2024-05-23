/*
 * CUDA Kernel for bit shuffling
 */

#include "cudabitshuffle.hpp"

#define CHUNK_SIZE 8192

// Define the pixel type
using pixel_t = H5Read::image_type;

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

// __global__ void cuda_bitshuffle(unsigned int *d_input, unsigned int
// *d_output,
//                                 int numElements, int numBits) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < numElements) {
//     unsigned int input = d_input[idx];
//     unsigned int output = 0;
//     for (int i = 0; i < numBits; i++) {
//       output |= ((input >> i) & 1) << (numBits - 1 - i);
//     }
//     d_output[idx] = output;
//   }
// }

/**
 * @brief: Swap the bytes of a 64-bit integer in place
 * @param: ptr - pointer to the 64-bit integer
 */
void byteswap64(void *ptr) {
  uint8_t *bytes = (uint8_t *)ptr;
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
}

/**
 * @brief: Swap the bytes of a 32-bit integer in place
 * @param: ptr - pointer to the 32-bit intege
 */
void byteswap32(void *ptr) {
  uint8_t *bytes = (uint8_t *)ptr;
  uint8_t tmp;
  tmp = bytes[0];
  bytes[0] = bytes[3];
  bytes[3] = tmp;
  tmp = bytes[1];
  bytes[1] = bytes[2];
  bytes[2] = tmp;
}

template <typename T>
__global__ void print_array_kernel(T *d_array, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    printf("%d: %f\n", idx, static_cast<float>(d_array[idx]));
  }
}

template <> __global__ void print_array_kernel<int>(int *d_array, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    printf("%d: %d\n", idx, d_array[idx]);
  }
}

template <>
__global__ void print_array_kernel<double>(double *d_array, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    printf("%d: %lf\n", idx, d_array[idx]);
  }
}

template <typename T> void print_array(T *d_array, size_t size) {
  // Define block size and grid size
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  // Launch kernel to print the array
  print_array_kernel<<<gridSize, blockSize>>>(d_array, size);

  // Synchronize device to ensure all prints are done
  cuda_throw_error();
  cudaDeviceSynchronize();
}

/**
 * @brief Get the absolute block offsets from the compressed chunk data
 * as well as the size of each compressed chunk. We use absolute offsets
 * so that we can easily calculate the pointer to each block on the GPU
 * @param h_buffer The compressed data
 * @param block_offsets The calculated absolute block offsets
 * @param block_sizes The calculated sizes of each block
 * @param batch_size The number of blocks
 */
void get_block_size_and_offset(uint8_t *h_buffer, size_t *block_offsets,
                               size_t *block_sizes, size_t batch_size) {
  // Byteswap the header
  byteswap64(h_buffer);
  byteswap32(h_buffer + 8);

  // Now byte swap the block headers
  uint8_t *block = h_buffer + 12; // Skip header
  printf("Block: %p\n", block);
  // print first 24 bytes of the block
  for (int i = 0; i < 24; i++) {
    printf("%d ", block[i]);
  }
  printf("\n");

  uint32_t image_size = (uint32_t) * (uint64_t *)h_buffer; // Get the image size
  uint32_t n_block = image_size / CHUNK_SIZE; // Calculate the number of blocks
  if (image_size % CHUNK_SIZE) { // If there is a remainder, add one more block
    n_block++;
  }

  // Ensure we do not exceed the allocated batch_size
  if (n_block > batch_size) {
    printf("Error: Number of blocks exceeds batch size.\n");
    return;
  }

  int cumulative_offset = 4;            // The cumulative offset of the blocks
  block_offsets[0] = cumulative_offset; // The first block starts at 0

  for (int i = 0; i < n_block; i++) {   // Iterate over the blocks
    byteswap32(block);                  // Byteswap the block header
    uint32_t next = *(uint32_t *)block; // Get the size of the block

    block_sizes[i] = next; // Add the size of the block to the block sizes
    cumulative_offset += next + 4; // Accumulate the offset
    block_offsets[i + 1] =
        cumulative_offset; // Add the offset to the block offsets
    block += next + 4;     // Move to the next block

    if (i < 10 || i > 4400) {
      printf("%d: Next: %d, Cumulative: %d, Block: %p, Block offset size: %d\n",
             i, next, cumulative_offset, block, i + 2);
    }
  }

  // Print the sizes of block_offsets and block_sizes
  printf("Block offsets size: %zu\n", n_block + 1);
  printf("Block sizes size: %zu\n", n_block);
}

/**
 * @brief Kernel to convert block offsets to gpu pointers, this replaces
 * the need for a for loop and more optimally utilises the GPU
 * @param d_data The compressed data on the device
 * @param d_block_offsets The offsets of the compressed blocks
 * @param d_ptr_list The output pointers to the compressed blocks
 * @param batch_size The number of blocks
 */
__global__ void block_offset_to_pointers_kernel(const uint8_t *d_data,
                                                const size_t *d_block_offsets,
                                                void **d_ptr_list,
                                                int batch_size) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < batch_size) {
    // Initialize the pointer to the beginning of the compressed data
    const uint8_t *current_ptr = d_data;

    // Move the pointer to the block offset
    current_ptr += d_block_offsets[i];

    if (i < 10 || i > 4400) {
      printf("Current pointer %d: %p   ", i, current_ptr);
      // // print first 24 bytes of the block
      // for (int i = 0; i < 24; i++) {
      //   printf("%d ", current_ptr[i]);
      // }
      // printf("\n");
    }

    // Set the pointer to the compressed block
    d_ptr_list[i] = (void *)current_ptr;
  }
}

/**
 * @brief Converts the block offsets to gpu memory pointers
 * @param d_data The compressed data on the device
 * @param block_offsets The offsets of the compressed blocks
 * @param d_compressed_ptrs The output pointers to the compressed blocks
 */
void block_offset_to_pointers(const uint8_t *d_data, size_t *d_block_offsets,
                              int batch_size, void **d_compressed_ptrs) {
  // Launch kernel
  dim3 blocks((batch_size + 255) /
              256); // Create enough blocks to cover all the indices
  dim3 threads(256);
  block_offset_to_pointers_kernel<<<blocks, threads>>>(
      d_data, d_block_offsets, d_compressed_ptrs, batch_size);
  cuda_throw_error();
  cudaDeviceSynchronize();
  // Free device memory
  cudaFree(d_block_offsets);
  printf("\n");
}

__global__ void print_pointers_kernel(void **d_uncompressed_ptrs,
                                      size_t batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size) {
    printf("Pointer %d: %p\n", idx, d_uncompressed_ptrs[idx]);
  }
}

__global__ void prefix_sum_kernel(size_t *d_uncompressed_bytes,
                                  size_t *d_prefix_sum_bytes,
                                  size_t *batch_size) {
  // Calculate the global thread ID
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= *batch_size) {
    return;
  }

  // Initialize the prefix sum
  size_t prefix_sum = 0;

  // Iterate over the uncompressed sizes
  for (size_t i = 0; i < idx; ++i) {
    // Add the current size to the prefix sum
    prefix_sum += d_uncompressed_bytes[i];
  }

  // Set the prefix sum for the current chunk
  d_prefix_sum_bytes[idx] = prefix_sum;
}

/**
 * @brief Decompresses the data using bitshuffle and LZ4 on the GPU
 */
void bshuf_decompress_lz4_gpu(uint8_t *h_compressed_data,
                              const size_t image_size, uint8_t *out) {
  int image_size_bytes = image_size * sizeof(pixel_t);

  // Print the first 24 bytes of the compressed data
  for (int i = 0; i < 24; i++) {
    printf("%d ", h_compressed_data[i]);
  }
  printf("\n");

  // Allocate device memory for the compressed data and copy from host to device
  uint8_t *d_compressed_data;
  cudaMalloc(&d_compressed_data, image_size_bytes);
  cudaMemcpy(d_compressed_data, h_compressed_data + 12, image_size_bytes,
             cudaMemcpyHostToDevice);
  printf("d_buffer size: %zu\n", image_size_bytes);

  // Set up CUDA stream for asynchronous operations
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Determine the number of blocks (batch size)
  size_t chunk_size = CHUNK_SIZE;
  size_t batch_size = (image_size_bytes + chunk_size - 1) / chunk_size + 1;

  printf("Batch size: %zu\n", batch_size);

  // Calculate block offsets and sizes from compressed data
  size_t *managed_block_offsets;
  size_t *managed_compressed_bytes;

  cudaMallocManaged(&managed_block_offsets, batch_size * sizeof(size_t));
  cudaMallocManaged(&managed_compressed_bytes, batch_size * sizeof(size_t));

  get_block_size_and_offset(h_compressed_data, managed_block_offsets,
                            managed_compressed_bytes, batch_size);

  // Allocate device memory for pointers to compressed and uncompressed data
  void **d_compressed_ptrs;
  // size_t *d_compressed_bytes;
  size_t *d_uncompressed_bytes;
  void **d_uncompressed_ptrs;
  cudaMalloc(&d_compressed_ptrs, sizeof(uint8_t *) * batch_size);
  // cudaMalloc(&d_compressed_bytes, sizeof(size_t) * batch_size);
  cudaMalloc(&d_uncompressed_bytes, sizeof(size_t) * batch_size);
  cudaMalloc(&d_uncompressed_ptrs, sizeof(size_t) * batch_size);

  // Print the first 10 block sizes
  for (int i = 0; i < 10; i++) {
    printf("Block size %d: %d\n", i, managed_compressed_bytes[i]);
  }

  print_array(managed_compressed_bytes, 10);

  // Convert block offsets to pointers on the GPU
  block_offset_to_pointers(d_compressed_data, managed_block_offsets, batch_size,
                           d_compressed_ptrs);

  // Decompression size and temporary buffer setup
  size_t decomp_temp_bytes;
  nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size,
                                        &decomp_temp_bytes);
  printf("Decompression temp size: %zu\n", decomp_temp_bytes);
  void *d_decomp_temp;
  cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

  // Setup for decompression error handling
  nvcompStatus_t *device_statuses;
  cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * batch_size);

  printf("Getting decompressed size\n");
  // Get the size of the decompressed data asynchronously
  nvcompBatchedLZ4GetDecompressSizeAsync(
      d_compressed_ptrs, managed_compressed_bytes, d_uncompressed_bytes,
      batch_size, stream);

  cudaStreamSynchronize(stream);

  printf("Uncompressed sizes\n");
  print_array(d_uncompressed_bytes, 10);
  print_array(d_uncompressed_bytes + batch_size - 10, 10);
  printf("\n");

  // Calculate the uncompressed pointers from the sizes
  size_t *d_batch_size;
  size_t *d_prefix_sum_bytes; // List of absolute offsets
  cudaMalloc(&d_prefix_sum_bytes, batch_size * sizeof(size_t));
  cudaMalloc(&d_batch_size, sizeof(size_t));
  cudaMemcpy(d_batch_size, &batch_size, sizeof(size_t), cudaMemcpyHostToDevice);
  dim3 blocks((batch_size + 255) / 256);
  dim3 threads(256);
  prefix_sum_kernel<<<blocks, threads>>>(d_uncompressed_bytes,
                                         d_prefix_sum_bytes, d_batch_size);
  cuda_throw_error();
  cudaDeviceSynchronize();

  block_offset_to_pointers((uint8_t *)d_uncompressed_ptrs, d_prefix_sum_bytes,
                           batch_size, d_uncompressed_ptrs);

  // Perform the decompression
  printf("Decompressing\n");
  nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
      d_compressed_ptrs, managed_compressed_bytes, d_uncompressed_bytes,
      nullptr, batch_size, d_decomp_temp, decomp_temp_bytes,
      d_uncompressed_ptrs, device_statuses, stream);

  // Check results of the decompression
  if (decomp_res != nvcompSuccess) {
    printf("Error in decompression\n");
  } else {
    printf("Decompression successful\n");
  }

  // Synchronize stream to ensure all operations are complete
  cudaStreamSynchronize(stream);

  // Check decompression status for each block
  nvcompStatus_t *host_statuses = new nvcompStatus_t[batch_size];
  cudaMemcpy(host_statuses, device_statuses,
             batch_size * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost);
  for (int i = 0; i < batch_size; ++i) {
    if (host_statuses[i] != nvcompSuccess) {
      printf("Decompression error on block %d: %d\n", i, host_statuses[i]);
    }
  }

  cudaStreamSynchronize(stream);

  print_array(d_uncompressed_bytes, 10);

  // Cleanup
  // delete[] host_statuses;
  cudaFree(d_compressed_data);
  cudaFree(d_compressed_ptrs);
  // cudaFree(d_compressed_bytes);
  cudaFree(d_uncompressed_bytes);
  cudaFree(d_uncompressed_ptrs);
  cudaFree(d_decomp_temp);
  cudaFree(device_statuses);
  cudaStreamDestroy(stream);
}