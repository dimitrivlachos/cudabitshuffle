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

void byteswap64(void *ptr);
void byteswap32(void *ptr);

void bshuf_decompress_lz4_gpu(uint8_t *h_compressed_data,
                              const size_t image_size, uint8_t *out);