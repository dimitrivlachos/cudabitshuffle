
#include "cudabitshuffle.hpp"

#define CHUNK_SIZE 8192

using pixel_t = H5Read::image_type;

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
    printf("n_block: %d, batch_size: %zu, image_size: %zu\n", n_block,
           batch_size, image_size);
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

    // if (i < 10 || i > 4400) {
    //   printf("%d: Next: %d, Cumulative: %d, Block: %p, Block offset size:
    //   %d\n",
    //          i, next, cumulative_offset, block, i + 2);
    // }
  }

  // Print the sizes of block_offsets and block_sizes
  printf("Block offsets size: %zu\n", n_block + 1);
  printf("Block sizes size: %zu\n", n_block);
}

__global__ void block_offsets_to_pointers_kernel(const uint8_t *d_data_ptr,
                                                 size_t *d_block_offsets,
                                                 void **d_ptrs,
                                                 size_t batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the index
  if (idx < batch_size) { // Ensure we do not exceed the batch size
    // Calculate the pointer to the block on the device by adding the block
    // offset to the data pointer
    d_ptrs[idx] = (void *)(d_data_ptr + d_block_offsets[idx]);
  }
}

/**
 * @brief Convert block offsets to pointers on the GPU
 * @param d_data_ptr Pointer to the data on the device
 * @param d_block_offsets Pointer to the block offsets on the device
 * @param d_ptrs (Output) Pointer to the pointers to the data on the device
 * @param batch_size The number of blocks
 * @param stream The CUDA stream to use
 */
void block_offsets_to_pointers(const uint8_t *d_data_ptr,
                               size_t *d_block_offsets, void **d_ptrs,
                               size_t batch_size, cudaStream_t stream) {

  dim3 blocks((batch_size + 255) / 256);
  dim3 threads(256);
  block_offsets_to_pointers_kernel<<<blocks, threads, 0, stream>>>(
      d_data_ptr, d_block_offsets, d_ptrs, batch_size);
  cuda_throw_error();
  cudaStreamSynchronize(stream);
}

__global__ void cumulative_sum_kernel(size_t *d_size_list,
                                      size_t *d_cumulative_size_list,
                                      size_t batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the index
  if (idx < batch_size) {            // Ensure we do not exceed the size
    size_t sum = 0;                  // Initialize the sum
    for (int i = 0; i <= idx; i++) { // Iterate over the elements
      sum += d_size_list[i];         // Add the element to the sum
    }
    d_cumulative_size_list[idx] = sum; // Set the output to the sum
  }
}

/**
 * @brief Calculate the cumulative sum of the sizes of the compressed data
 * @param d_size_list Pointer to the sizes of the compressed data on the device
 * @param d_cumulative_size_list (Output) Pointer to the cumulative sum of the
 * sizes of the compressed data on the device
 * @param batch_size The number of blocks
 * @param stream The CUDA stream to use
 */
void cumulative_sum(size_t *d_size_list, size_t *d_cumulative_size_list,
                    size_t batch_size, cudaStream_t stream) {
  dim3 blocks((batch_size + 255) / 256);
  dim3 threads(256);
  cumulative_sum_kernel<<<blocks, threads, 0, stream>>>(
      d_size_list, d_cumulative_size_list, batch_size);
  cuda_throw_error();
  cudaStreamSynchronize(stream);
}

/**
 * @brief Seperates and decompresses bitshuffle-standard LZ4 compressed
 * data chunks.
 * @param h_compressed_data Pointer to the compressed data on the host.
 * @param image_size Size of the image in pixels.
 * @param d_decompressed_data Pointer to the decompressed data on the
 * device.
 */
void nvcomp_decompress_lz4(uint8_t *h_compressed_data, const size_t image_size,
                           uint8_t *d_decompressed_data) {
  int image_size_bytes = image_size * sizeof(pixel_t);

  // Allocate device memory for the compressed data
  // Then copy the compressed data to the device
  uint8_t *d_compressed_data; // Device pointer to the compressed data
  cudaMalloc(&d_compressed_data, image_size_bytes);
  cudaMemcpy(d_compressed_data, h_compressed_data, image_size_bytes,
             cudaMemcpyHostToDevice);

  // Set up CUDA stream for asynchronous operations
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Determine the number of blocks (batch size)
  size_t chunk_size = CHUNK_SIZE;
  size_t batch_size = (image_size_bytes + chunk_size - 1) / chunk_size + 1;

  // Calculate block offsets and sizes from compressed data
  size_t *managed_compressed_block_offsets;
  size_t *managed_compressed_bytes;
  cudaMallocManaged(&managed_compressed_block_offsets,
                    batch_size * sizeof(size_t));
  cudaMallocManaged(&managed_compressed_bytes, batch_size * sizeof(size_t));
  get_block_size_and_offset(h_compressed_data, managed_compressed_block_offsets,
                            managed_compressed_bytes, batch_size);

  // Allocate device memory for pointers to compressed and uncompressed data
  void **d_compressed_ptrs; // Device pointer to compressed data chunk pointers
  void **
      d_uncompressed_ptrs; // Device pointer to uncompressed data chunk pointers
  size_t *d_uncompressed_bytes; // Device pointer to uncompressed data sizes
  cudaMalloc(&d_compressed_ptrs, sizeof(uint8_t *) * batch_size);
  cudaMalloc(&d_uncompressed_ptrs, sizeof(size_t) * batch_size);
  cudaMalloc(&d_uncompressed_bytes, sizeof(size_t) * batch_size);

  // Convert block offsets to pointers on the GPU
  block_offsets_to_pointers(d_compressed_data, managed_compressed_block_offsets,
                            d_compressed_ptrs, batch_size, stream);

  // Set up temporary size and buffer for decompression
  size_t decomp_temp_bytes; // Size of the temporary buffer for decompression
  void *
      d_decomp_temp; // Device pointer to the temporary buffer for decompression
  nvcompBatchedLZ4DecompressGetTempSize(
      batch_size, chunk_size,
      &decomp_temp_bytes); // Get the size of the temporary buffer
  cudaMalloc(&d_decomp_temp,
             decomp_temp_bytes); // Allocate memory for the temporary buffer

  // Get the size of the decompressed data
  nvcompBatchedLZ4GetDecompressSizeAsync(
      d_compressed_ptrs, managed_compressed_bytes, d_uncompressed_bytes,
      batch_size, stream);

  cudaStreamSynchronize(stream); // Synchronize the stream

  // Calculate the uncompressed pointer offsets
  size_t *d_decompressed_block_offsets; // Device pointer to the cumulative
                                        // sizes of the uncompressed data
  cudaMalloc(&d_decompressed_block_offsets,
             sizeof(size_t) * batch_size); // Allocate memory for the cumulative
                                           // sizes of the uncompressed data
  cumulative_sum(
      d_uncompressed_bytes, d_decompressed_block_offsets, batch_size,
      stream); // Calculate the cumulative sizes of the uncompressed data

  // Convert block offsets to pointers on the GPU
  block_offsets_to_pointers(d_decompressed_data, d_decompressed_block_offsets,
                            d_uncompressed_ptrs, batch_size, stream);

  // Set up decompression error statuses
  nvcompStatus_t
      *d_statuses; // Device pointer to the decompression error statuses
  cudaMalloc(
      &d_statuses,
      sizeof(nvcompStatus_t) *
          batch_size); // Allocate memory for the decompression error statuses

  /* === Decompress === */
  nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
      d_compressed_ptrs, managed_compressed_bytes, d_uncompressed_bytes,
      nullptr, batch_size, d_decomp_temp, decomp_temp_bytes,
      d_uncompressed_ptrs, d_statuses, stream);

  cudaStreamSynchronize(stream); // Synchronize the stream

  // Check for decompression errors
  if (decomp_res != nvcompSuccess) {
    printf("Error: Decompression failed.\n");
    return;
  }

  // Free device memory
  cudaFree(d_compressed_data);
  cudaFree(d_compressed_ptrs);
  cudaFree(d_uncompressed_ptrs);
  cudaFree(d_uncompressed_bytes);
  cudaFree(d_decomp_temp);
  cudaFree(d_statuses);
  cudaFree(managed_compressed_block_offsets);
  cudaFree(managed_compressed_bytes);
  cudaFree(d_decompressed_block_offsets);
}