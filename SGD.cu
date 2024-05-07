/*
    A program trying to implement SGD using CUDA.
    Target function: Y = F(X) = (x1^t1 + x2^t2 + ... + x16^t16)/16 (t_i \in [0.5, 2]) (x_i \in [0, 1])
    Network Structure: Input -- Linear -- ReLU -- Linear -- Sigmoid -- Average
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <exception>

#include "assert.h"
#include "stdlib.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "./layers/linear.hh"
#include "./layers/relu.hh"

cublasHandle_t handle;

class Sigmoid {

};

void cuda_sgd

void randomize_matrix() {

}

void data_generation() {
    
}



int main() {
    float lr = 0.1;
    int input_size = 16;
    int hidden_size = {16, 16};
    int output_size = 1;

    A = (float *)malloc(sizeof(float) * input_size * hidden_size[0] * batch_size);
    B = (float *)malloc(sizeof(float) * hidden_size[0] * hidden_size[1] * batch_size);
    C = (float *)malloc(sizeof(float) * hidden_size[1] * output_size * batch_size);

    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };


    return 0;
}