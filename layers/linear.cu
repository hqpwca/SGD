#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear.hh"
#include "../utils/exception.hh"

Linear::Linear(Shape shape) : W(shape), b(shape.y, 1) { //Input, Output
    this->type = TLinear;
    
    b.allocateMemory();
    for (int x = 0; x < b.shape.x; x++)
        b[x] = 0;
    b.copyHostToDevice();

    W.allocateMemory();
    std::default_random_engine generator;
    std::normal_distribution<float> normal_distribution(0.0, 1.0);
    for (int x = 0; x < W.shape.x; x++)
        for (int y = 0; y < W.shape.y; y++)
            W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
    W.copyHostToDevice();
}

Linear::~Linear()
{ }

__global__ void updateBiasKernel(float *bias, float *grad, int batch_size, int num_output, float learning_rate) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_output) {
        float sum = 0.0;
        for (int i = 0; i < batch_size; i++) {
            sum += grad[idx + i * num_output];
        }
        bias[idx] -= learning_rate * sum / batch_size;
    }
}

Matrix& Linear::forward(cublasHandle_t &cublasH, Matrix &x) { // x: (num_input, batch_size)
    assert(x.shape.x == W.shape.x);

    Shape Y_shape(W.shape.y, x.shape.y);
    Y.allocateMemoryIfNotAllocated(Y_shape);
    this->X = x;

    const float alpha = 1.0, beta = 0.0;

    CUBLAS_CHECK( \
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, \
                    W.shape.y, x.shape.y, W.shape.x, &alpha, \
                    W.data_device.get(), W.shape.x, \ 
                    x.data_device.get(), W.shape.x, \
                    &beta, Y.data_device.get(), W.shape.y) \
                );

    const float alpha_add = 1.0, beta_add = 1.0;

    for (int i = 0; i < x.shape.y; ++i) {
        CUBLAS_CHECK(
            cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                        W.shape.y, 1, &alpha_add,
                        b.data_device.get(), W.shape.y,
                        &beta_add, Y.data_device.get() + i * W.shape.y, W.shape.y,
                        Y.data_device.get() + i * W.shape.y, W.shape.y)
        );
    }
    
    return Y;
}

Matrix& Linear::back_prop(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01) { // od: (num_output, batch_size)
    assert(od.shape.x == W.shape.y);

    Shape d_shape(W.shape.x, od.shape.y); // d: (num_input, batch_size)  d = W * od
    d.allocateMemoryIfNotAllocated(d_shape);

    const float alpha = 1.0, beta = 0.0;

    CUBLAS_CHECK( \
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, \
                    W.shape.x, od.shape.y, W.shape.y, &alpha, \
                    W.data_device.get(), W.shape.x, \ 
                    od.data_device.get(), W.shape.y, \
                    &beta, d.data_device.get(), W.shape.x) \
                );

    updateWeights(cublasH, od, lr);
    updateBias(cublasH, od, lr);

    return d;
}

void Linear::updateWeights(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01) {  // od: (num_output, batch_size)
    const float alpha = lr / od.shape.y, beta = 1.0;

    CUBLAS_CHECK( \
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, \
                    X.shape.x, od.shape.x, X.shape.y, &alpha, \
                    X.data_device.get(), X.shape.x, \ 
                    od.data_device.get(), od.shape.y, \
                    &beta, W.data_device.get(), W.shape.x) \
                );
}

void Linear::updateBias(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01) {  // od: (num_output, batch_size)
    int blockSize = 32;
    int numBlocks = (od.shape.x + blockSize - 1) / blockSize;

    updateBiasKernel<<<numBlocks, blockSize>>>(b.data_device.get(), od.data_device.get(), od.shape.y, od.shape.x, lr);

    NNException::throwIfDeviceErrorsOccurred("Cannot perform bias update.");
}