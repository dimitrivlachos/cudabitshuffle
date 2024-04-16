# CUDA Bitshuffle
CUDA based implementation of the bitshuffle algorithm.

This will allow you to compress and decompress data using the bitshuffle algorithm on the GPU, negating the need to transfer data between the CPU and GPU. This is very useful for large, streamed and compressed datasets such as data transferred through hdf5 files, which often use the bitshuffle algorithm along with another compression algorithm such as LZ4.

# References
This is a direct implementation of the bitshuffle algorithm, which is described in the original repository: https://github.com/kiyo-masui/bitshuffle
It is meant to be fully compatible with the original implementation, and should be able to compress and decompress data in the same way.