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

class cuda_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

inline auto cuda_error_string(cudaError_t err) {
  const char *err_name = cudaGetErrorName(err);
  const char *err_str = cudaGetErrorString(err);
  printf("CUDA Error: %s: %s\n", err_name, err_str);
}

/// Raise an exception IF CUDA is in an error state, with the name and
/// description
inline auto cuda_throw_error() -> void {
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s: %s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
  } else {
    printf("No CUDA error\n");
  }
}

void byteswap64(void *ptr);
void byteswap32(void *ptr);

void nvcomp_decompress_lz4(uint8_t *h_compressed_data, const size_t image_size,
                           uint8_t *d_decompressed_data);

void bshuf_untrans_bit_elem_CUDA(const void *in, void *out, size_t size,
                                 size_t elem_size);
