#include "cudabitshuffle.hpp"
#include "h5read.h"
#include <cuda_runtime.h>
#include <iostream>

using pixel_t = H5Read::image_type;

int main() {
  H5Read reader("../data/ins_13_1_1.nxs");

  int height = reader.image_shape()[0];
  int width = reader.image_shape()[1];

  pixel_t *host_image = new pixel_t[width * height];
  pixel_t *host_results = new pixel_t[width * height];

  // auto host_image = cudaMallocPitch(&host_image, width * sizeof(pixel_t),
  // height); auto host_results = cudaMallocPitch(&host_results, width *
  // sizeof(pixel_t), height);

  // Buffer for reading compressed chunk data in
  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

  auto buffer = reader.get_raw_chunk(0, raw_chunk_buffer);

  // Print the first 10 elements of the compressed chunk data
  for (int i = 0; i < 100; i++) {
    std::cout << (int)raw_chunk_buffer[i] << " ";
  }

  // // Decompress the chunk data
  // auto decompressed_chunk_buffer = make_cuda_pinned_malloc<pixel_t>(width *
  // height); bshuf_decompress_lz4(
  //     raw_chunk_buffer.data(), decompressed_chunk_buffer.get(), width *
  //     height, 2, 0);

  // // Bitshuffle the decompressed data
  // bshuf_bitshuffle(
  //     decompressed_chunk_buffer.get(), host_results.get(), width, height,
  //     sizeof(pixel_t));

  // bshuf_decompress_lz4(
  //        buffer.data() + 12, host_image.get(), width * height, 2, 0);
  // TODO:
  // 1. Read compressed chunk data from the file
  // 2. Decompress the chunk data
  // 3. Bitshuffle the decompressed data
  // 4. Copy the bitshuffled data to the host_results buffer
  // 5. Compare the bitshuffled data with the original data
  // Need to do this with original bshuffle and gpu version ^

  return 0;
}