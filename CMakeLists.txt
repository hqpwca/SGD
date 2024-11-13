cmake_minimum_required(VERSION 3.10)
project(SGD LANGUAGES CXX CUDA)
enable_language(CUDA)

SET(BUILD_MODE Debug)
SET(CMAKE_CUDA_ARCHITECTURES 86)

# Specify include directories
add_library(network SHARED
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
    layers/A_B0.cu
    layers/A_B0.hh
    layers/A_B1.cu
    layers/A_B1.hh
    layers/A_H0.cu
    layers/A_H0.hh
    layers/A_Hl.cu
    layers/A_Hl.hh
    layers/Ref_projector.hh
    layers/Ref_projector.cu
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

target_compile_options(network PRIVATE -fopenmp $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(network PRIVATE -lcublas -fopenmp)

# add_executable(SGD SGD.cu)

# set_property(TARGET SGD 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# target_compile_options(SGD PRIVATE -fopenmp $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
# target_link_libraries(SGD PRIVATE network -lcublas -fopenmp)

add_executable(Bilinear Bilinear.cu)

set_property(TARGET Bilinear 
            PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(Bilinear PRIVATE -fopenmp $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(Bilinear PRIVATE network -lcublas -fopenmp)