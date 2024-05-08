#pragma once

#include "../utils/matrix.hh"

class Linear {
    const float weights_init_threshold = 0.01;

    Matrix W; // (num_input, num_output)
    Matrix b; // (num_output, batch_size)

    Matrix Y; // (num_output, batch_size)
    Matrix X; // (num_input, batch_size)
    Matrix d; // (num_input, batch_size)
public:
    Linear(Shape shape) : W(shape), b(shape.y, 1) { //Input, Output
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
    ~Linear();

    Matrix& forward(cublasHandle_t &cublasH, Matrix &x); // x: (num_input, batch_size)
    Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od); // od: (num_output, batch_size)
    void updateWeights(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01); // od: (num_output, batch_size)
};