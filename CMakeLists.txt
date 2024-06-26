cmake_minimum_required(VERSION 3.10)
project(SGD LANGUAGES CXX CUDA)
enable_language(CUDA)

SET(BUILD_MODE Debug)
SET(CMAKE_CUDA_ARCHITECTURES 86)

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
    layers/Ref_projector.cu
    layers/Ref_projector.hh
    utils/dataset.cu
    utils/dataset.hh
    utils/matrix.cu
    utils/matrix.hh
    utils/matrix_double.cu
    utils/matrix_double.hh
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

target_compile_options(network PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(SGD SGD.cu)

set_property(TARGET SGD 
            PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(SGD PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(SGD PRIVATE network -lcublas)