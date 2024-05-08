#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <cmath>

#include "sigmoid.hh"
#include "../utils/exception.hh"

Sigmoid::Sigmoid() { this->type = TSigmoid; }

Sigmoid::~Sigmoid() { }

//TODO: Do multiple calcualtion in each kernel.

__global__ void sigmoidKernel(float* Y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        Y[index] = 1.0f / (1.0f + expf(-Y[index]));
    }
}

__global__ void sigmoidBackpropKernel(const float* Y, float* dX, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        dX[index] = dX[index] * Y[index] * (1.0f - Y[index]);
    }
}

Matrix& Sigmoid::forward(cublasHandle_t &cublasH, Matrix &x) {
    this->X = x;
    Y.allocateMemoryIfNotAllocated(x.shape);

    int num_elements = x.shape.x * x.shape.y;

    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    cudaMemcpy(Y.data_device.get(), x.data_device.get(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);

    sigmoidKernel<<<numBlocks, blockSize>>>(Y.data_device.get(), num_elements);
    cudaDeviceSynchronize();

    NNException::throwIfDeviceErrorsOccurred("Cannot perform Sigmoid forward propagation.");

    return Y;
}

Matrix& Sigmoid::back_prop(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01) {
    d.allocateMemoryIfNotAllocated(od.shape);

    int num_elements = X.shape.x * X.shape.y;

    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    cudaMemcpy(d.data_device.get(), od.data_device.get(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);

    sigmoidBackpropKernel<<<numBlocks, blockSize>>>(Y.data_device.get(), d.data_device.get(), num_elements);
    cudaDeviceSynchronize();

    NNException::throwIfDeviceErrorsOccurred("Cannot perform Sigmoid back propagation");

    return d;
}