#include "cudabitshuffle.hpp"
#include "h5read.h"
#include <cuda_runtime.h>
#include <iostream>

// Define the pixel type
using pixel_t = H5Read::image_type;

int main() {
  H5Read reader("../data/ins_13_1_1.nxs");

  int height = reader.image_shape()[0];
  int width = reader.image_shape()[1];

  pixel_t *host_image = new pixel_t[width * height];
  pixel_t *host_results = new pixel_t[width * height];

  // Buffer for reading compressed chunk data in
  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

  auto buffer = reader.get_raw_chunk(0, raw_chunk_buffer);

  std::cout << "Buffer size: " << buffer.size() << std::endl;

  // Print the first 50 elements of the compressed chunk data
  for (int i = 0; i < 50; i++) {
    std::cout << (int)raw_chunk_buffer[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}