#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear.hh"
#include "../utils/exception.hh"

Linear::~Linear()
{ }

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

Matrix& Linear::back_prop(cublasHandle_t &cublasH, Matrix &od) { // od: (num_output, batch_size)
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
