#include "cudabitshuffle.hpp"
#include "common.hpp"
#include "h5read.h"
#include <bitshuffle.h>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <iostream>

// Define the pixel type
using pixel_t = H5Read::image_type;

class cuda_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

inline auto cuda_error_string(cudaError_t err) {
  const char *err_name = cudaGetErrorName(err);
  const char *err_str = cudaGetErrorString(err);
  return fmt::format("{}: {}", std::string{err_name}, std::string{err_str});
}

/// Allocate memory using cudaMallocHost
template <typename T> auto make_cuda_pinned_malloc(size_t num_items = 1) {
  using Tb = typename std::remove_extent<T>::type;
  Tb *obj = nullptr;
  auto err = cudaMallocHost(&obj, sizeof(Tb) * num_items);
  if (err != cudaSuccess || obj == nullptr) {
    throw cuda_error(fmt::format("Error in make_cuda_pinned_malloc: {}",
                                 cuda_error_string(err)));
  }
  auto deleter = [](Tb *ptr) { cudaFreeHost(ptr); };
  return std::shared_ptr<T[]>{obj, deleter};
}

int main() {
  H5Read reader("../data/ins_13_1_1.nxs");

  int height = reader.image_shape()[0];
  int width = reader.image_shape()[1];

  auto host_decompressed_image =
      make_cuda_pinned_malloc<pixel_t>(width * height);

  // Buffer for reading compressed chunk data in
  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

  // Create empty buffer to store the compressed chunk data
  SPAN<uint8_t> buffer;

  buffer = reader.get_raw_chunk(49, raw_chunk_buffer);

  // Print the first 50 elements of the compressed chunk data
  for (int i = 0; i < 50; i++) {
    std::cout << (int)buffer[i] << " ";
  }
  std::cout << std::endl;

  // Check chunk compression type
  auto compression = reader.get_raw_chunk_compression();
  std::cout << "Chunk compression: " << compression << std::endl;

  // Decompress and deshuffle the data using the bitshuffle library
  bshuf_decompress_lz4(buffer.data() + 12, host_decompressed_image.get(),
                       width * height, 2, 0);

  int j = 0;
  while (host_decompressed_image[j] == 0) {
    j++;
  }
  std::cout << "J: " << j << std::endl;
  for (int i = 0; i < 50; i++) {
    std::cout << (int)host_decompressed_image[460 + i] << " ";
  }
  std::cout << std::endl;

  // image data x, y, width and height, image width and height
  draw_image_data(host_decompressed_image.get(), 1640, 1690, 35, 40, width,
                  height);

  /* Perform and compare cuda bitshuffle */

  // Allocate memory on the device

  return 0;
}