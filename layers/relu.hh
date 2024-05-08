#pragma once

#include "../utils/matrix.hh"

class ReLU {
    Matrix Y; // (_, batch_size)
    Matrix X; // (_, batch_size)
    Matrix d; // (_, batch_size)
public:
    ReLU();
    ~ReLU();

    Matrix& forward(cublasHandle_t &cublasH, Matrix &X);
    Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od);
};