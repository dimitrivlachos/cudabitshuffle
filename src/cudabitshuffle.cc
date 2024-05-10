#include "cudabitshuffle.hpp"

// Define the pixel type
using pixel_t = H5Read::image_type;

// void byteswap64(void *ptr) {
//   uint8_t *bytes = (uint8_t *)ptr;
//   uint8_t tmp;
//   tmp = bytes[0];
//   bytes[0] = bytes[7];
//   bytes[7] = tmp;
//   tmp = bytes[1];
//   bytes[1] = bytes[6];
//   bytes[6] = tmp;
//   tmp = bytes[2];
//   bytes[2] = bytes[5];
//   bytes[5] = tmp;
//   tmp = bytes[3];
//   bytes[3] = bytes[4];
//   bytes[4] = tmp;
// }

// void byteswap32(void *ptr) {
//   uint8_t *bytes = (uint8_t *)ptr;
//   uint8_t tmp;
//   tmp = bytes[0];
//   bytes[0] = bytes[3];
//   bytes[3] = tmp;
//   tmp = bytes[1];
//   bytes[1] = bytes[2];
//   bytes[2] = tmp;
// }

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

template <typename T> struct PitchedMalloc {
public:
  using value_type = T;
  PitchedMalloc(std::shared_ptr<T[]> data, size_t width, size_t height,
                size_t pitch)
      : _data(data), width(width), height(height), pitch(pitch) {}

  PitchedMalloc(size_t width, size_t height) : width(width), height(height) {
    auto [alloc, alloc_pitch] = make_cuda_pitched_malloc<T>(width, height);
    _data = alloc;
    pitch = alloc_pitch;
  }

  auto get() { return _data.get(); }
  auto size_bytes() -> size_t const { return pitch * height * sizeof(T); }
  auto pitch_bytes() -> size_t const { return pitch * sizeof(T); }

  std::shared_ptr<T[]> _data;
  size_t width;
  size_t height;
  size_t pitch;
};

void print_bytes(void *buffer, int length) {
  uint8_t *bytes = (uint8_t *)buffer;
  for (int i = 0; i < length; i++) {
    std::cout << (int)bytes[i] << " ";
  }
  std::cout << std::endl;
}

/**
 * @brief: Swap the bytes of a 64-bit integer in place
 * @param: ptr - pointer to the 64-bit integer
 */
void byteswap64(const void *ptr) {
  printf("Byteswap64\n");
  // Print first 64 bytes of the buffer
  print_bytes((void *)ptr, 64);
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
  // Print first 64 bytes of the buffer
  print_bytes((void *)ptr, 64);
}

/**
 * @brief: Swap the bytes of a 32-bit integer in place
 * @param: ptr - pointer to the 32-bit intege
 */
void byteswap32(void *ptr) {
  // printf("Byteswap32\n");
  // Print first 32 bytes of the buffer
  // print_bytes(ptr, 32)
  uint8_t *bytes = (uint8_t *)ptr;
  uint8_t tmp;
  tmp = bytes[0];
  bytes[0] = bytes[3];
  bytes[3] = tmp;
  tmp = bytes[1];
  bytes[1] = bytes[2];
  bytes[2] = tmp;
  // Print first 32 bytes of the buffer
  // print_bytes(ptr, 32);
}

std::vector<int> get_absolute_block_offsets(uint8_t *buffer) {
  std::vector<int> block_offsets;

  // Byteswap the header
  byteswap64(buffer);
  byteswap32(buffer + 8);

  // Now byte swap the block headers
  // printf("Buffer: %p\n", buffer);
  // print_bytes(buffer, 20);
  uint8_t *block = buffer + 12; // Skip the header
  // printf("Block: %p\n", block);
  // print_bytes(block, 20);
  uint32_t image_size = (uint32_t) * (uint64_t *)buffer;
  // printf("Image size: %d\n", image_size);
  uint32_t n_block = image_size / 8192;
  // printf("Number of blocks: %d\n", n_block);
  if (image_size % 8192)
    n_block++;
  // fmt::print(fg(fmt::color::yellow), "Begin block offsets\n");
  int cumulative_offset = 0;                  // First block starts at 0
  block_offsets.push_back(cumulative_offset); // First block starts at 0
  for (int i = 0; i < n_block; i++) {
    // if (i < 10) {
    //   printf("Block: %p\n", block);
    //   print_bytes(block, 20);
    // }
    // print_bytes(block, 20);
    byteswap32(block);
    // printf("Block byteswapped32\n");
    // print_bytes(block, 20);
    // printf("\n");
    uint32_t next = *(uint32_t *)block;
    auto dist_to_next = std::distance(block, block + next + 4);
    cumulative_offset += dist_to_next;
    if (cumulative_offset > 1000000000)
      printf("Cumulative offset: %d\n", cumulative_offset);
    block_offsets.push_back(cumulative_offset);
    // fmt::print("Block offset: {}\n", next);
    block += next + 4;
  }

  return block_offsets;
}

void cpu_decompress(H5Read *reader, std::shared_ptr<pixel_t[]> *out,
                    int chunk_index) {
  // Get the image width and height
  int height = reader->image_shape()[0];
  int width = reader->image_shape()[1];

  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

  SPAN<uint8_t> buffer;
  buffer = reader->get_raw_chunk(chunk_index, raw_chunk_buffer);

  // Print the first 50 elements of the compressed chunk data
  for (int i = 0; i < 50; i++) {
    std::cout << (int)buffer[i] << " ";
  }
  std::cout << std::endl;

  // Get the chunk compression type
  auto compression = reader->get_raw_chunk_compression();
  std::cout << "Chunk compression: " << compression << std::endl;

  // Decompress and deshuffle the data using the bitshuffle library
  bshuf_decompress_lz4(buffer.data() + 12, out->get(), width * height, 2, 0);

  int j = 0;
  while ((*out)[j] == 0) {
    j++;
  }
  std::cout << "J: " << j << std::endl;

  // Print 50 elements around the first non-zero element of the decompressed
  // data
  for (int i = 0; i < 50; i++) {
    // std::cout << (int)(*out)[j - 25 + i] << " ";
    std::cout << (int)(*out)[460 + i] << " ";
  }
  std::cout << std::endl;
}

void gpu_decompress(H5Read *reader, auto *out, int chunk_index) {
  // Get the image width and height
  int height = reader->image_shape()[0];
  int width = reader->image_shape()[1];

  printf("Width: %d, Height: %d, pixel_t: %d\n", width, height,
         sizeof(pixel_t));
  printf("Size: %d\n", width * height * sizeof(pixel_t));

  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

  SPAN<uint8_t> buffer;
  buffer = reader->get_raw_chunk(chunk_index, raw_chunk_buffer);

  uint8_t *d_buffer;
  int length = width * height * sizeof(pixel_t);
  cudaMalloc(&d_buffer, length);

  // Print the size of buffer
  std::cout << "Buffer size: " << buffer.size_bytes() << " bytes" << std::endl;

  // Print the first 12 bytes of buffer.data()
  for (int i = 0; i < 12; i++) {
    std::cout << (int)buffer[i] << " ";
  }
  std::cout << std::endl;

  // Make a copy of buffer for testing
  std::vector<uint8_t> vbuffer_copy(buffer.begin(), buffer.end());
  SPAN<uint8_t> buffer_copy(vbuffer_copy);

  printf("Buffer bytes\n");
  // Print the first 12 bytes of buffer.data()
  for (int i = 0; i < 12; i++) {
    std::cout << (int)buffer[i] << " ";
  }
  printf("\n");
  // Print the first 12 bytes of buffer_copy.data()
  for (int i = 0; i < 12; i++) {
    std::cout << (int)buffer_copy[i] << " ";
  }
  printf("\n\n");

  // byte swap the header
  // byteswap64(buffer.data());
  // byteswap32(buffer.data() + 8);

  // now byte swap the block headers
  // uint8_t * block = buffer.data() + 12;
  // uint32_t image_size = (uint32_t)*(uint64_t *)buffer.data();
  // uint32_t n_block = image_size / 8192;
  // if (image_size % 8192) n_block ++;
  // for (int i = 0; i < n_block; i++) {
  //   if (i < 10) {
  //     printf("Block: %p\n", block);
  //   print_bytes(block, 20);
  //   }
  //   byteswap32(block);
  //   uint32_t next = *(uint32_t *) block;
  //   block += next + 4;
  // }

  // Copy the compressed chunk data to the device
  cudaMemcpy(d_buffer, buffer.data() + 12, buffer.size_bytes(),
             cudaMemcpyHostToDevice);

  print_array(d_buffer, length, 0);

  // Create output buffer
  uint8_t *d_out;
  cudaMalloc(&d_out, width * height * sizeof(pixel_t));

  // Get the block offsets
  printf("\n\nMy byteswap\n");
  auto absolute_block_offsets = get_absolute_block_offsets(buffer.data());
  for (int i = 0; i < 10; i++) {
    fmt::print("Block offset: {}\n", absolute_block_offsets[i]);
  }

  decompress_lz4_gpu(d_buffer, length, d_out, width * height * sizeof(pixel_t),
                     absolute_block_offsets);

  // Get the chunk compression type
  auto compression = reader->get_raw_chunk_compression();
  std::cout << "Chunk compression: " << compression << std::endl;
}

int main() {
  H5Read reader("../data/ins_13_1_1.nxs");

  int height = reader.image_shape()[0];
  int width = reader.image_shape()[1];

  // auto host_decompressed_image =
  //     make_cuda_pinned_malloc<pixel_t>(width * height);

  auto device_decompressed_image =
      make_cuda_pitched_malloc<pixel_t>(width, height);

  // fmt::print("CPU Decompression\n");

  // cpu_decompress(&reader, &host_decompressed_image, 49);

  // // image data x, y, width and height, image width and height
  // draw_image_data(host_decompressed_image.get(), 1640, 1690, 35, 40, width,
  //                 height);

  fmt::print("GPU Decompression\n");

  gpu_decompress(&reader, &device_decompressed_image, 49);

  return 0;
}