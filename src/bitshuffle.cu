#include "cudabitshuffle.hpp"

/**
 * @brief Transpose bytes in a matrix where each row represents a bit of the
 * elements.
 *
 * This kernel transposes the bits in the input data such that each row of bits
 * is converted into a byte in the output data. The input is organized in a way
 * that each row represents a specific bit of the elements.
 *
 * @param in Input data array, each element treated as a byte (char).
 * @param out Output data array where the transposed result will be stored.
 * @param nrows Number of rows in the bit matrix, calculated as 8 * elem_size.
 * @param nbyte_row Number of bytes in each row of the input data, calculated as
 * size / 8.
 *
 * Each thread operates on one element of the matrix and performs the following
 * steps:
 * 1. Loads 8 bytes from each row.
 * 2. Performs bit interleaving to transpose the bits.
 * 3. Writes the transposed bits to the output array.
 *
 * Example:
 * Given the input bytes:
 * in: [0b11001010, 0b01101001, 0b10101100, 0b00101111, 0b01010101, 0b11110000,
 * 0b00011100, 0b10010110] The kernel transposes the bits to produce the output
 * bytes.
 */
__global__ void transpose_byte_bitrow(const char *in, char *out, size_t nrows,
                                      size_t nbyte_row) {
  size_t ii = blockIdx.x * blockDim.x + threadIdx.x; // Row index
  size_t jj = blockIdx.y * blockDim.y + threadIdx.y; // Column index

  if (ii >= nrows || jj >= nbyte_row) {
    // Return if the thread is outside the array bounds
    return;
  }
  // Load 8 bytes from each row
  char row_data[8];
  for (int k = 0; k < 8; ++k) {
    row_data[k] = in[(ii + k) * nbyte_row + jj];
  }

  // Perform bit interleaving
  for (int bit = 0; bit < 8; ++bit) {
    char result = 0;
    for (int byte = 0; byte < 8; ++byte) {
      /*
       * Extract the bit-th bit from row_data[byte] by right-shifting
       * 'byte' by 'bit' positions and masking with 1. This isolates
       * the desired bit from the current byte.
       *
       * The extracted bit is then shifted to its new position in the
       * 'result' variable, which accumulates the bits for the current
       * position from all 8 bytes.
       *
       * The operation ((row_data[byte] >> bit) & 1) isolates the bit-th
       * bit from row_data[byte], and << byte shifts it to its correct
       * position in the result byte.
       */
      result |= ((row_data[byte] >> bit) & 1) << byte;
    }
    out[jj * nrows + ii + bit] = result;
  }
}

/**
 * @brief Launches the transpose_byte_bitrow CUDA kernel.
 *
 * This function configures the grid and block dimensions and launches the
 * transpose_byte_bitrow kernel to perform bit transposition on the input data.
 * It also synchronizes the device to ensure all threads have completed before
 * returning control to the host.
 *
 * @param in Pointer to the input data array.
 * @param out Pointer to the output data array where the transposed result will
 * be stored.
 * @param size Size of the input data.
 * @param elem_size Size of each element in the data.
 *
 * The function calculates the number of rows (nrows) and bytes per row
 * (nbyte_row) and sets up the CUDA grid and block dimensions. It then launches
 * the transpose_byte_bitrow kernel and waits for its completion.
 */
void launch_transpose_byte_bitrow(const void *in, void *out, size_t size,
                                  size_t elem_size) {
  size_t nrows = 8 * elem_size;
  size_t nbyte_row = size / 8;

  dim3 block_size(16, 16);
  dim3 grid_size((nrows + block_size.x - 1) / block_size.x,
                 (nbyte_row + block_size.y - 1) / block_size.y);

  transpose_byte_bitrow<<<grid_size, block_size>>>(
      (const char *)in, (char *)out, nrows, nbyte_row);
  cudaDeviceSynchronize();
}

/**
 * @brief Shuffle bits within the bytes of blocks of eight elements.
 *
 * This kernel shuffles bits within the bytes of eight-element blocks. It
 * operates by loading the input data into shared memory, extracting bits using
 * the `__ballot_sync` intrinsic, and repositioning them in the output array.
 *
 * @param in Input data array, each element treated as a byte (char).
 * @param out Output data array where the shuffled bits will be stored, each
 * element treated as a 16-bit integer.
 * @param nbyte Total number of bytes to process, calculated as elem_size *
 * size.
 * @param elem_size Size of each element in the data.
 *
 * Each thread operates on a specific byte and bit position within the data and
 * performs the following steps:
 * 1. Loads bytes from the input data into shared memory for efficient access.
 * 2. Uses the `__ballot_sync` intrinsic to create bitmasks from the bits across
 * a warp.
 * 3. Repositions the bits across the blocks and stores the result in the output
 * array.
 *
 * Example:
 * Given the input bytes:
 * in: [0b11001010, 0b01101001, 0b10101100, 0b00101111, 0b01010101, 0b11110000,
 * 0b00011100, 0b10010110] The kernel shuffles the bits within the bytes and
 * stores the shuffled result in the output array.
 */
__global__ void shuffle_bit_eightelem(const char *in, uint16_t *out,
                                      size_t nbyte, size_t elem_size) {
  size_t ii = blockIdx.x * blockDim.x + threadIdx.x; // Byte index
  size_t jj = blockIdx.y * blockDim.y + threadIdx.y; // Bit index

  if (ii >= nbyte || jj >= 8 * elem_size) {
    // Return if the thread is outside the array bounds
    return;
  }
  /*
   * We use shared memory to load 8 bytes from the input data.
   * Each thread in the block loads one byte from the input data.
   * The shared memory block_data is a 2D array with 8 rows and 16 columns.
   * Each row corresponds to a thread in the block, and each column
   * corresponds to a byte. The block_data array is used to load 8 bytes from
   * the input data in a coalesced manner. The __syncthreads() function is
   * called to synchronize all threads in the block.
   */
  __shared__ char block_data[8][16];

  if (threadIdx.y < 16) {
    block_data[threadIdx.x][threadIdx.y] = in[ii + jj + threadIdx.y];
  }
  __syncthreads();

  /*
   * This portion only executes for the first 8 threads in the block.
   * As these are the threads responsible for shuffling the bits within the
   * bytes.
   */
  if (threadIdx.y < 8) {
    int32_t bt; // Holds the bit to be shuffled
    char xmm =
        block_data[threadIdx.x][threadIdx.y]; // Holds the byte to be shuffled

    for (int k = 0; k < 8; ++k) {
      bt = __ballot_sync(0xFFFFFFFF,
                         (xmm & 1) != 0); // Ballot the least significant bit
      xmm >>= 1; // Shift the byte to the right by 1 bit to prepare for the
                 // next bit
      size_t ind =
          ii + (jj / 8) + (7 - k) * elem_size; // Calculate the output index
      if (threadIdx.y == 0) { // Only the first thread writes the result
        out[ind / 2] = bt;
      }
    }
  }
}

/**
 * @brief Launches the shuffle_bit_eightelem CUDA kernel.
 *
 * This function configures the grid and block dimensions and launches the
 * shuffle_bit_eightelem kernel to perform bit shuffling on the input data. It
 * also synchronizes the device to ensure all threads have completed before
 * returning control to the host.
 *
 * @param in Pointer to the input data array.
 * @param out Pointer to the output data array where the shuffled bits will be
 * stored.
 * @param size Size of the input data.
 * @param elem_size Size of each element in the data.
 *
 * The function calculates the total number of bytes to process (nbyte) and sets
 * up the CUDA grid and block dimensions. It then launches the
 * shuffle_bit_eightelem kernel and waits for its completion.
 */
void launch_shuffle_bit_eightelem(const void *in, void *out, size_t size,
                                  size_t elem_size) {
  size_t nbyte = elem_size * size;

  dim3 block_size(8, 8);
  dim3 grid_size((nbyte + block_size.x - 1) / block_size.x,
                 (8 * elem_size + block_size.y - 1) / block_size.y);

  shuffle_bit_eightelem<<<grid_size, block_size>>>(
      (const char *)in, (uint16_t *)out, nbyte, elem_size);
  cudaDeviceSynchronize();
}

/**
 * @brief De-shuffles and untransposes bits within elements in CUDA.
 *
 * This function performs de-shuffling and untransposition of bits within
 * elements. It combines two CUDA kernels: one for transposing bytes within bit
 * rows and another for shuffling bits within the bytes of eight-element blocks.
 *
 * @param in Pointer to the input data array.
 * @param out Pointer to the output data array where the de-shuffled and
 * untransposed result will be stored.
 * @param size Size of the input data.
 * @param elem_size Size of each element in the data.
 *
 * The function performs the following steps:
 * 1. Allocates temporary buffer on the device to hold intermediate results.
 * 2. Calls the `launch_transpose_byte_bitrow` function to transpose bytes
 * within bit rows.
 * 3. Calls the `launch_shuffle_bit_eightelem` function to shuffle bits within
 * the bytes of eight-element blocks.
 * 4. Frees the temporary buffer.
 *
 * Example:
 * Given the input data, this function will first transpose the bytes within bit
 * rows and then shuffle the bits within the bytes of eight-element blocks,
 * storing the final result in the output array.
 */
void bshuf_untrans_bit_elem_CUDA(const void *in, void *out, size_t size,
                                 size_t elem_size) {
  void *tmp_buf;
  cudaMalloc(&tmp_buf, size * elem_size);

  // Step 1: Transpose bytes within bit rows
  launch_transpose_byte_bitrow(in, tmp_buf, size, elem_size);

  // Step 2: Shuffle bits within bytes of eight-element blocks
  launch_shuffle_bit_eightelem(tmp_buf, out, size, elem_size);

  cudaFree(tmp_buf);
}
