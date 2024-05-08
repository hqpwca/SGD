#pragma once

#include "../utils/matrix.hh"
#include "layer.hh"

class Linear : public Layer{
    const float weights_init_threshold = 0.01;

    Matrix W; // (num_input, num_output)  W_T (num_output, num_input)
    Matrix b; // (1, batch_size)

    Matrix Y; // (num_output, batch_size)
    Matrix X; // (num_input, batch_size)
    Matrix d; // (num_input, batch_size)
public:
    Linear(Shape shape);
    ~Linear();

    Matrix& forward(cublasHandle_t &cublasH, Matrix &x); // x: (num_input, batch_size)
    Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01); // od: (num_output, batch_size)
    void updateWeights(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01); // od: (num_output, batch_size)
    void updateBias(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01); // od: (num_output, batch_size)
};