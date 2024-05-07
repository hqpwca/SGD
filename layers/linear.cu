#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear.hh"
#include "../utils/exception.hh"

Linear::Linear(Shape shape); : W(shape), b(shape.y, 1) { //Input, Output
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

Matrix& Linear::forward(cublasHandle_t &cublasH, Matrix &x) { // x: (num_input, batch_size)
    assert(x.shape.x == W.shape.x);

    Shape Y_shape(W.shape.y, x.shape.y);
    Y.allocateMemoryIfNotAllocated(Y_shape);
    this->X = x;

    CUBLAS_CHECK( \
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, \
                    W.shape.y, x.shape.y, W.shape.x, 1.0, \
                    W.data_device.get(), W.shape.x, \ 
                    x.data_device.get(), W.shape.x, \
                    0.0, Y.data_device.get(), W.shape.y) \
                );

    // TODO: add bias
    
    return Y;
}

Matrix& Linear::back_prop(cublasHandle_t &cublasH, Matrix &od) { // od: (num_output, batch_size)
    assert(od.shape.x == W.shape.y);

    Shape d_shape(W.shape.x, od.shape.y); // d: (num_input, batch_size)  d = W * od
    d.allocateMemoryIfNotAllocated(d_shape);

    CUBLAS_CHECK( \
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, \
                    W.shape.x, od.shape.y, W.shape.y, 1.0, \
                    W.data_device.get(), W.shape.x, \ 
                    od.data_device.get(), W.shape.y, \
                    0.0, d.data_device.get(), W.shape.x) \
                );

    return d;
}

void Linear::updateWeights(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01) {  // od: (num_output, batch_size)
    CUBLAS_CHECK( \
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, \
                    X.shape.x, od.shape.x, X.shape.y, lr / od.shape.y, \
                    X.data_device.get(), X.shape.x, \ 
                    od.data_device.get(), od.shape.y, \
                    0.0, W.data_device.get(), W.shape.x) \
                );
}
