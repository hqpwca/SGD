cmake_minimum_required(VERSION 3.10)
enable_language(CUDA)

project(SGD LANGUAGES CXX CUDA)

SET(BUILD_MODE Debug)

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
    layers/SF_projector.cu
    layers/SF_projector.hh
    utils/dataset.cu
    utils/dataset.hh
    utils/matrix.cu
    utils/matrix.hh
    utils/shape.cu
    utils/shape.hh
    utils/geo.cu
    utils/geo.hh
    )

target_compile_features(network PUBLIC cxx_std_14)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( network
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(network PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)

add_executable(SGD SGD.cu)

set_property(TARGET SGD 
            PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(SGD PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)
target_link_libraries(SGD PRIVATE network -lcublas)