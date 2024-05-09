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
#include "utils/dataset.hh"

cublasHandle_t handle;
std::vector<float> T;
Dataset *train, *test;

Matrix calcSquareLoss(Matrix &batch_output, Matrix &network_output) // (1, batch_size)
{
    assert(batch_output.shape.x == network_output.shape.x && batch_output.shape.y == network_output.shape.y);

    Matrix m(batch_output.shape);
    for (int i = 0; i < batch_output.shape.x * batch_output.shape.y; ++ i) {
        m[i] =  std::pow((network_output[i] - batch_output[i]), 2.0);
    }
    return m;
}

float sumLoss(Matrix &m) {
    float sum = 0.0;
    for (int i = 0; i < m.shape.x * m.shape.y; ++ i) {
        sum += m[i];
    }
    return sum;
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
    delete train;
    delete test;
}

void train_SGD(Network N, Dataset *d, float lr = 0.1, int epoches = 100) {
    for (int epoch = 0; epoch < epoches; ++epoch) {
        d->nextEpoch();

        int num_batches = -1;
        float sumloss = 0.0;
        while(d->nextBatch()) {
            ++ num_batches;
            Matrix input = d->getBatchInput();
            Matrix output = d->getBatchOutput();

            //Forward
            Matrix net_output = N.forward(handle, input);
            net_output.copyDeviceToHost();

            //Calc loss
            Matrix loss = calcSquareLoss(output, net_output);
            sumloss += sumLoss(loss);
            loss.copyHostToDevice();

            //Back prop
            N.back_prop(handle, loss, lr);
        }
        std::cerr << "Finished Epoch #" << epoch << " Loss: " << sumloss / num_batches / 32 << std::endl;
    }
}

float test_SGD(Network N, Dataset *d) {

}

int main(int argc, char *argv[]) {
    Network N;
    init_network(N);

    train = new Dataset(T, 16, 1, 32, 2048);
    test = new Dataset(T, 16, 1, 32, 512);

    train_SGD(N, train);

    cleanup();

    return 0;
}