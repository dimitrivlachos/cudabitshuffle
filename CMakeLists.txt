cmake_minimum_required(VERSION 3.20)
project(cudabitshuffle LANGUAGES CXX CUDA)

# Set the C++ standard to use
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)
include(UseSpanBackportIfNeeded)

# Add the directories
include_directories(include)
include_directories(common/include)

# Add subdirectories
add_subdirectory(h5read)
add_subdirectory(dependencies/fmt)

add_library(lodepng STATIC dependencies/lodepng/lodepng.cpp)
target_include_directories(lodepng INTERFACE dependencies/lodepng)
target_compile_definitions(lodepng INTERFACE HAVE_LODEPNG)

# Set path to nvcomp
set(nvcomp_DIR "${CMAKE_SOURCE_DIR}/ENV/lib/cmake/nvcomp")

# Find the required packages
find_package(nvcomp REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)

# Add the executable
add_executable(
    cudabitshuffle
    src/cudabitshuffle.cc
    src/cuda_utils.cu
    src/nv_decompress.cu
    src/bitshuffle.cu
)

# Add dependencies
target_link_libraries(
    cudabitshuffle
    PRIVATE
    fmt
    lodepng
    h5read
    CUDA::cudart
    LZ4::LZ4
    Bitshuffle::bitshuffle
    nvcomp::nvcomp
)
target_compile_options(
    cudabitshuffle
    PRIVATE
    "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>"
)