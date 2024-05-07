#pragma once

#include "../utils/matrix.hh"

class ReLU {
    Matrix Y; // (num_output, batch_size)
    Matrix X; // (num_input, batch_size)
    Matrix d; // (num_input, batch_size)
public:
    ReLU();
    ~ReLU();

    Matrix& forward(cublasHandle_t &cublasH, Matrix &X);
    Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od);
};