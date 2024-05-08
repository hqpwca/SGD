#pragma once

#include "../utils/matrix.hh"
#include "layer.hh"

class Sigmoid : public Layer {
    Matrix Y; // (num_output, batch_size)
    Matrix X; // (num_input, batch_size)
    Matrix d; // (num_input, batch_size)
public:
    Sigmoid();
    ~Sigmoid();

    Matrix& forward(cublasHandle_t &cublasH, Matrix &X);
    Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01);
};