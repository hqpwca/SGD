#pragma once

#include <cuda_runtime.h>
#include "layer.hh"
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class Bilinear : public Layer{
    int nblock, ngrid;
public:
    GeoData *geodata;

    Bilinear(GeoData *geo);
    ~Bilinear();

    void project(Matrix &vol, Matrix &proj, double weight);
    void project(Matrix &vol, MatrixD &proj, double weight);
    void matrix(MatrixD &mat);
    void back_project(Matrix &vol, Matrix &proj, double weight);

    Matrix& forward(cublasHandle_t &cublasH, Matrix &x); // x: (num_input, batch_size)
    Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01); // od: (num_output, batch_size)
};

extern "C" {
    void bilinear_matrix_generation(int np, int nu, int nx, int ny, double dx, double dy, double du, double lsd, double lso, double *nmat);
}