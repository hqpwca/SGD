/*
    A program trying to implement SGD using CUDA.
    Target function: Y = F(X) = (x1^t1 + ((x1+x2)/2)^t2 + ... + ((x1+...+x16)/16)^t16)/16 (t_i \in [0.5, 2]) (x_i \in [0, 1])
    Network Structure: Input -- Linear(16, 32) -- ReLU -- Linear(32, 16) -- ReLU -- Linear(16, 1) -- Sigmoid
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
#include "./layers/sigmoid.hh"

cublasHandle_t handle;

void data_generation() {
    
}

int main() {
    float lr = 0.1;

    Linear L1 = Linear(Shape(16, 32));
    ReLU R1 = ReLU();
    Linear L2 = Linear(Shape(32, 16));
    ReLU R2 = ReLU();
    Linear L3 = Linear(Shape(16, 1));

    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    data_generation();



    cublasDestroy(handle);

    return 0;
}