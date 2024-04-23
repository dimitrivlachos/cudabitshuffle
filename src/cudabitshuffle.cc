#include "cudabitshuffle.hpp"
#include "h5read.h"
#include <bitshuffle.h>
#include <cuda_runtime.h>
#include <iostream>

// Define the pixel type
using pixel_t = H5Read::image_type;

/// Allocate memory using cudaMallocHost
template <typename T> auto make_cuda_pinned_malloc(size_t num_items = 1) {
  using Tb = typename std::remove_extent<T>::type;
  Tb *obj = nullptr;
  auto err = cudaMallocHost(&obj, sizeof(Tb) * num_items);
  if (err != cudaSuccess || obj == nullptr) {
    throw std::runtime_error("Failed to allocate pinned memory");
  }
  auto deleter = [](Tb *ptr) { cudaFreeHost(ptr); };
  return std::shared_ptr<T[]>{obj, deleter};
}

int main() {
  H5Read reader("../data/ins_13_1_1.nxs");

  int height = reader.image_shape()[0];
  int width = reader.image_shape()[1];

  // pixel_t *host_image = new pixel_t[width * height];
  // pixel_t *host_results = new pixel_t[width * height];

  auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);
  auto host_results = make_cuda_pinned_malloc<uint8_t>(width * height);

  // Buffer for reading compressed chunk data in
  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

  // int number_of_images = reader.get_number_of_images();

  // Create empty buffer to store the compressed chunk data
  SPAN<uint8_t> buffer;

  buffer = reader.get_raw_chunk(50, raw_chunk_buffer);

  // for (int i = 0; i < number_of_images; i++) {
  //   if (reader.is_image_available(i)) {
  //     auto buffer = reader.get_raw_chunk(i, raw_chunk_buffer);
  //     std::cout << "Buffer size: " << buffer.size() << std::endl;
  //     break;
  //   }
  // }

  // std::cout << "Buffer size: " << buffer.size() << std::endl;

  // Print the first 50 elements of the compressed chunk data
  for (int i = 0; i < 50; i++) {
    std::cout << (int)buffer[i] << " ";
  }
  std::cout << std::endl;

  // Check chunk compression type
  auto compression = reader.get_raw_chunk_compression();
  std::cout << "Chunk compression: " << compression << std::endl;

  // Decompress and deshuffle the data using the bitshuffle library
  bshuf_decompress_lz4(buffer.data() + 12, host_image.get(), width * height, 2,
                       0);

  int j = 0;
  while (host_image[j] == 0) {
    j++;
  }
  std::cout << "J: " << j << std::endl;
  for (int i = 0; i < 50; i++) {
    std::cout << (int)host_image[460 + i] << " ";
  }
  std::cout << std::endl;

  return 0;
}