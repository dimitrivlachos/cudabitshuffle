#include "cudabitshuffle.hpp"
#include "h5read.h"
#include <iostream>

template <typename T>
auto make_cuda_pitched_malloc(size_t width, size_t height) {
  static_assert(!std::is_unbounded_array_v<T>,
                "T automatically returns unbounded array pointer");
  size_t pitch = 0;
  T *obj = nullptr;
  auto err = cudaMallocPitch(&obj, &pitch, width * sizeof(T), height);
  if (err != cudaSuccess || obj == nullptr) {
    throw cuda_error(fmt::format("Error in make_cuda_pitched_malloc: {}",
                                 cuda_error_string(err)));
  }

  auto deleter = [](T *ptr) { cudaFree(ptr); };

  return std::make_pair(std::shared_ptr<T[]>(obj, deleter), pitch / sizeof(T));
}

int main() {
  H5Read reader("../data/ins_13_1_1.nxs");

  int height = reader.image_shape()[0];
  int width = reader.image_shape()[1];

  auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);
  auto host_results = make_cuda_pinned_malloc<uint8_t>(width * height);

  // Buffer for reading compressed chunk data in
  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

  // Read the compressed chunk data from the file
  reader.read_chunk_data(raw_chunk_buffer.data(),
                         width * height * sizeof(pixel_t));

  // Print the first 10 elements of the compressed chunk data
  for (int i = 0; i < 10; i++) {
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