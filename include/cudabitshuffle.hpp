#pragma once

#include "h5read.h"
#include <cstdint>
#include <cuda_runtime.h>

void run_test();

void print_array(uint8_t *d_buffer, int length, int index);