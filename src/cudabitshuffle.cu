/*
 * CUDA Kernel for bit shuffling
*/

__global__ void cuda_bitshuffle(unsigned int *d_input, unsigned int *d_output, int numElements, int numBits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int input = d_input[idx];
        unsigned int output = 0;
        for (int i = 0; i < numBits; i++) {
            output |= ((input >> i) & 1) << (numBits - 1 - i);
        }
        d_output[idx] = output;
    }
}