#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "relu.hh"
#include "../utils/exception.hh"

ReLU:ReLU() { }

ReLU:~ReLU() { }

Matrix& ReLU::forward(cublasHandle_t &cublasH, Matrix &x) { //(_, batch_size)
    this.X = x;
    Y.allocateMemoryIfNotAllocated(x.shape);

    CUBLAS_CHECK(cublasIsamax(cublasH, ));

    return Y;
}

Matrix& ReLU::back_prop(cublasHandle_t &cublasH, Matrix &od) { //(_, batch_size)

    return d;
}