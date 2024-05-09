#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear.hh"
#include "../utils/exception.hh"

Linear::Linear(Shape shape) : W(shape), b(1, shape.y) { //Input, Output (columns, rows);
    this->type = TLinear;
    
    b.allocateMemory();
    for (int x = 0; x < b.shape.y; x++)
        b[x] = 0;
    b.copyHostToDevice();

    W.allocateMemory();
    std::default_random_engine generator;
    std::normal_distribution<float> normal_distribution(0.0, 1.0);
    for (int y = 0; y < W.shape.y; y++)
        for (int x = 0; x < W.shape.x; x++)
            W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
    W.copyHostToDevice();
}

Linear::~Linear()
{ }

__global__ void updateBiasKernel(float *bias, float *grad, int batch_size, int num_output, float learning_rate) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size) {
        for (int i = 0; i < num_output; i++)
            bias[idx] -= learning_rate * grad[idx * num_output + i] / batch_size;
    }
}

Matrix& Linear::forward(cublasHandle_t &cublasH, Matrix &x) { // x: (num_input, batch_size)
    assert(x.shape.x == W.shape.x);

    Shape Y_shape(W.shape.y, x.shape.y);
    Y.allocateMemoryIfNotAllocated(Y_shape); // Y: (num_output, batch_size)  Y = X * W_T
    this->X = x;

    const float alpha = 1.0, beta = 0.0;

    CUBLAS_CHECK(
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                    x.shape.y, W.shape.y, W.shape.x, &alpha,
                    x.data_device.get(), x.shape.y,
                    W.data_device.get(), W.shape.y,
                    &beta, Y.data_device.get(), Y.shape.y)
                );

    const float alpha_add = 1.0, beta_add = 1.0;

    for (int i = 0; i < x.shape.y; ++i) {
        CUBLAS_CHECK(
            cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                        1, Y.shape.x, &alpha_add,
                        b.data_device.get(), Y.shape.x, &beta_add,
                        Y.data_device.get() + i * W.shape.y, 1,
                        Y.data_device.get() + i * W.shape.y, 1)
        );
    }
    
    return Y;
}

Matrix& Linear::back_prop(cublasHandle_t &cublasH, Matrix &od, float lr) { // od: (num_output, batch_size)
    assert(od.shape.x == W.shape.y);

    Shape d_shape(W.shape.x, od.shape.y); // d: (num_input, batch_size)  d = od * W
    d.allocateMemoryIfNotAllocated(d_shape);

    const float alpha = 1.0, beta = 0.0;

    CUBLAS_CHECK( 
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                    od.shape.y, W.shape.x, od.shape.x, &alpha,
                    od.data_device.get(), od.shape.y, 
                    W.data_device.get(), W.shape.y, &beta,
                    d.data_device.get(), d.shape.y)
                );

    updateWeights(cublasH, od, lr);
    updateBias(cublasH, od, lr);

    return d;
}

void Linear::updateWeights(cublasHandle_t &cublasH, Matrix &od, float lr) {  // od: (num_output, batch_size)
    const float alpha = lr / od.shape.y, beta = 1.0;

    // W = W - lr * (od_T * X) / bs
    CUBLAS_CHECK( 
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                    W.shape.x, W.shape.y, X.shape.y, &alpha,
                    od.data_device.get(), od.shape.y, 
                    X.data_device.get(), X.shape.y, &beta,
                    W.data_device.get(), W.shape.x)
                );
}

void Linear::updateBias(cublasHandle_t &cublasH, Matrix &od, float lr) {  // od: (num_output, batch_size)
    int blockSize = 32;
    int numBlocks = (od.shape.y + blockSize - 1) / blockSize;

    updateBiasKernel<<<numBlocks, blockSize>>>(b.data_device.get(), od.data_device.get(), od.shape.y, od.shape.x, lr);

    NNException::throwIfDeviceErrorsOccurred("Cannot perform bias update.");
}