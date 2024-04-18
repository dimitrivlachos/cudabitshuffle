cmake_minimum_required(VERSION 3.20)
project(cudabitshuffle LANGUAGES CXX CUDA)

# Set the C++ standard to use
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# Find the CUDA package
find_package(CUDAToolkit REQUIRED)

# Add the directories
include_directories(include)

# Find the required packages
find_package(nvcomp REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Add the executable
add_executable(
    cudabitshuffle
    src/cudabitshuffle.cc
    src/cudabitshuffle.cu
)

# Add dependencies
target_link_libraries(
    cudabitshuffle
    PRIVATE
)
target_compile_options(
    cudabitshuffle
    PRIVATE
    "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>"
)