#pragma once

#define Z_SIZE 64

#include <cuda_runtime.h>
#include "layer.hh"
#include "../utils/matrix.hh"
#include "../utils/geo.hh"

class SF : public Layer{
    dim3 vblock, vgrid;
public:
    GeoData *geodata;

    SF(GeoData *geo);
    ~SF();

    void project(Matrix &vol, Matrix &proj, double weight);
    void back_project(Matrix &vol, Matrix &proj, double weight);

    Matrix& forward(cublasHandle_t &cublasH, Matrix &x); // x: (num_input, batch_size)
    Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01); // od: (num_output, batch_size)
};