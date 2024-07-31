#include "cudabitshuffle.hpp"
#include <lodepng.h>

// Define the pixel type
using pixel_t = H5Read::image_type;

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

void gpu_decompress(H5Read *reader, uint8_t *out, size_t pitch,
                    int chunk_index) {
  // Get the image width and height
  int height = reader->image_shape()[0];
  int width = reader->image_shape()[1];

  uint8_t *d_decompressed_image;
  cudaMallocPitch(&d_decompressed_image, &pitch, width * sizeof(pixel_t),
                  height);

  printf("Width: %d, Height: %d, pixel_t: %d\n", width, height,
         sizeof(pixel_t));
  printf("Size: %d\n", width * height * sizeof(pixel_t));

  auto raw_chunk_buffer =
      std::vector<uint8_t>(width * height * sizeof(pixel_t));

  SPAN<uint8_t> buffer;
  buffer = reader->get_raw_chunk(chunk_index, raw_chunk_buffer);

  // // Make a copy of buffer for testing
  // std::vector<uint8_t> vbuffer_copy(buffer.begin(), buffer.end());
  // SPAN<uint8_t> buffer_copy(vbuffer_copy);

  // printf("Buffer bytes\n");
  // // Print the first 12 bytes of buffer.data()
  // for (int i = 0; i < 24; i++) {
  //   std::cout << (int)buffer[i] << " ";
  // }
  // printf("\n");
  // // Print the first 12 bytes of buffer_copy.data()
  // for (int i = 0; i < 24; i++) {
  //   std::cout << (int)buffer_copy[i] << " ";
  // }
  // printf("\n\nMain byteswap\n");

  // // byte swap the header
  // byteswap64(buffer_copy.data());
  // byteswap32(buffer_copy.data() + 8);

  // // now byte swap the block headers
  // uint8_t *block = buffer_copy.data() + 12;
  // uint32_t image_size = (uint32_t) * (uint64_t *)buffer_copy.data();
  // uint32_t n_block = image_size / 8192;
  // printf("host n_block: %d, image_size: %d\n", n_block, image_size);
  // if (image_size % 8192)
  //   n_block++;
  // for (int i = 0; i < n_block; i++) {
  //   if (i < 10) {
  //     printf("Block: %p\n", block);
  //     print_bytes(block, 20);
  //   }
  //   byteswap32(block);
  //   uint32_t next = *(uint32_t *)block;
  //   block += next + 4;
  // }
  // printf("\n\n GPU byteswap\n");

  // bshuf_decompress_lz4_gpu(buffer.data(), width * height, out);
  // nvcomp_decompress_lz4(buffer.data() + 12, width * height, out);
  nvcomp_decompress_lz4(buffer.data(), width * height, d_decompressed_image);

  uint8_t *d_unshuffled_image;
  cudaMallocPitch(&d_unshuffled_image, &pitch, width * sizeof(pixel_t), height);

  // Perform bit unshuffling and transposing on the GPU
  bshuf_untrans_bit_elem_CUDA(d_decompressed_image, d_unshuffled_image,
                              width * height, sizeof(uint8_t));

  // Copy the decompressed image from the device to the host
  cudaMemcpy2D(out, width * sizeof(pixel_t), d_unshuffled_image, pitch,
               width * sizeof(pixel_t), height, cudaMemcpyDeviceToHost);

  // Get the chunk compression type
  auto compression = reader->get_raw_chunk_compression();
  std::cout << "Chunk compression: " << compression << std::endl;
}

int main() {
  H5Read reader("../data/ins_13_1_1.nxs");

  int height = reader.image_shape()[0];
  int width = reader.image_shape()[1];

  // auto device_decompressed_image =
  //     make_cuda_pitched_malloc<pixel_t>(width, height);

  // fmt::print("CPU Decompression\n");

  // cpu_decompress(&reader, &host_decompressed_image, 49);

  // // image data x, y, width and height, image width and height
  // draw_image_data(host_decompressed_image.get(), 1640, 1690, 35, 40, width,
  //                 height);

  fmt::print("GPU Decompression\n");

  // uint8_t *decompressed_image = new uint8_t[width * height *
  // sizeof(pixel_t)];
  uint8_t *h_decompressed_image = new uint8_t[width * height * sizeof(pixel_t)];
  // cudaMallocManaged(&h_decompressed_image, width * height * sizeof(pixel_t));

  gpu_decompress(&reader, h_decompressed_image,
                 width * height * sizeof(pixel_t), 49);
  // draw_image_data(decompressed_image, width-35, height-40, 35, 40, width,
  // height);

  // Check if there are non-zero elements in the decompressed image
  int j = 0;
  while (h_decompressed_image[j] == 0) {
    if (j > width * height * sizeof(pixel_t)) {
      std::cout << "No non-zero elements found" << std::endl;
      return 0;
    }
    j++;
  }
  std::cout << "J: " << j << std::endl;

  // Print 50 elements around the first non-zero element of the decompressed
  // data
  for (int i = 0; i < 50; i++) {
    std::cout << (int)h_decompressed_image[j - 25 + i] << " ";
  }
  std::cout << std::endl;

  // Assuming bytes, so uint8_t, calculate j's position in the image
  int x = j % width;
  int y = j / width;

  std::cout << "X: " << x << ", Y: " << y << std::endl;

  // print the size of the image
  std::cout << "Image size: " << width << " x " << height << std::endl;

  return 0;
}