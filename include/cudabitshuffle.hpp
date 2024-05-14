#pragma once

#include "common.hpp"
#include "h5read.h"
#include <bitshuffle.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <iostream>
#include <nvcomp/lz4.h>
#include <stdio.h>

void print_array(uint8_t *d_buffer, int length, int index);

void byteswap64(void *ptr);
void byteswap32(void *ptr);

void bshuf_decompress_lz4_gpu(uint8_t *h_compressed_data,
                              const size_t image_size);

void decompress_lz4_gpu(const uint8_t *compressed_data, size_t compressed_size,
                        uint8_t *decompressed_data, size_t decompressed_size,
                        const std::vector<int> &absolute_block_offsets);

// Define the pixel type
using pixel_t = H5Read::image_type;