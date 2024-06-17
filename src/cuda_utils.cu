#include "cudabitshuffle.hpp"

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

/**
 * @brief: Print the array on the device
 * @param: d_array - pointer to the device array
 * @param: size - size of the array
 */
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