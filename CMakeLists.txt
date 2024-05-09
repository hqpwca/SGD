cmake_minimum_required(VERSION 3.10)
enable_language(CUDA)

project(SGD LANGUAGES CXX CUDA)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Specify include directories
add_library(network STATIC
    network.cu
    network.hh
    layers/linear.cu
    layers/linear.hh
    layers/relu.cu
    layers/relu.hh
    layers/sigmoid.cu
    layers/sigmoid.hh
    layers/layer.hh
    utils/dataset.cu
    utils/dataset.hh
    utils/matrix.cu
    utils/matrix.hh
    utils/shape.cu
    utils/shape.hh
    )

target_compile_features(network PUBLIC cxx_std_14)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( network
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(SGD SGD.cu)

set_property(TARGET SGD 
            PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(SGD PRIVATE network -lcublas)