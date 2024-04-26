#pragma once

#include "common.hpp"
#include "h5read.h"
#include <bitshuffle.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <iostream>
#include <nvcomp.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <stdio.h>

void run_test();

void print_array(uint8_t *d_buffer, int length, int index);

void nvc_decompress(uint8_t *d_buffer);