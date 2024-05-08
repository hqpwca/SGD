#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "relu.hh"
#include "../utils/exception.hh"

ReLU::ReLU() { this->type = TReLU; }

ReLU::~ReLU() { }

__global__ void reluKernel(float* Y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        Y[index] = fmaxf(0.0f, Y[index]);
    }
}

__global__ void reluBackpropKernel(const float* Y, float* dX, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        dX[index] = Y[index] > 0 ? dX[index] : 0.0f;
    }
}

Matrix& ReLU::forward(cublasHandle_t &cublasH, Matrix &x) { //(_, batch_size)
    this->X = x;
    Y.allocateMemoryIfNotAllocated(x.shape);

    int num_elements = x.shape.x * x.shape.y;

    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    cudaMemcpy(Y.data_device.get(), x.data_device.get(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);

    reluKernel<<<numBlocks, blockSize>>>(Y.data_device.get(), num_elements);
    cudaDeviceSynchronize();

    NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU forward propagation.");

    return Y;
}

Matrix& ReLU::back_prop(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01) { //(_, batch_size)
    d.allocateMemoryIfNotAllocated(od.shape);

    int num_elements = X.shape.x * X.shape.y;

    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    cudaMemcpy(d.data_device.get(), od.data_device.get(), num_elements * sizeof(float), cudaMemcpyDeviceToDevice);

    reluBackpropKernel<<<numBlocks, blockSize>>>(X.data_device.get(), d.data_device.get(), num_elements);
    cudaDeviceSynchronize();

    NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU back propagation");

    return d;
}