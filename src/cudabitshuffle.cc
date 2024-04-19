#include "cudabitshuffle.h"
#include "h5read.h"
#include <iostream>

int main() {
  H5Read reader("../data/ins_13_1_1.nxs");

  int height = reader.image_shape()[0];
  int width = reader.image_shape()[1];

  auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);
  auto host_results = make_cuda_pinned_malloc<uint8_t>(width * height);

  // Buffer for reading compressed chunk data in
  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

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