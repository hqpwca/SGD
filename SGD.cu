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

#include "network.hh"

cublasHandle_t handle;

Matrix calcLoss(Matrix &batch_output, Matrix &network_output)
{
    
}

void init_network(Network &N) {
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    N.addLayer(new Linear(Shape(16, 32)));
    N.addLayer(new ReLU());
    N.addLayer(new Linear(Shape(32, 16)));
    N.addLayer(new ReLU());
    N.addLayer(new Linear(Shape(16, 1)));
    N.addLayer(new Sigmoid());
}

void cleanup() {
    cublasDestroy(handle);
}

void train(float lr = 0.1) {

}

float test() {

}

int main() {

    Network N;
    init_network(N);

    train();

    cleanup();

    return 0;
}