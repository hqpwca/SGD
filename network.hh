#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <exception>

#include "assert.h"
#include "stdlib.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "./layers/linear.hh"
#include "./layers/relu.hh"
#include "./layers/sigmoid.hh"

class Network {
    std::vector<Layer *> layers;

    Matrix output;
    Matrix grad;
    float lr;
public:
    Network(float lr = 0.01);
    ~Network();

    void addLayer(Layer *layer);
    std::vector<Layer *> getLayers() const;

    Matrix& forward(cublasHandle_t &cublasH, Matrix &x);
    void back_prop(cublasHandle_t &cublasH, Matrix &loss, float lr = 0.1);
};