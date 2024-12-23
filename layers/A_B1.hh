#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_B1{
    int nblock, ngrid;
    dim3 vblock, vgrid, vgrid_z;
    Matrix *vecs;

public:
    GeoData *geodata;

    A_B1(GeoData *geo);
    ~A_B1();

    void project(MatrixD &vol, MatrixD &proj, double weight);
    void back_project(MatrixD &vol, MatrixD &proj, double weight);
};

extern "C" {
    A_B1 *b1_init(int nx, int ny, int np, int nu, double dx, double dy, double du, double lsd, double lso, double *angles);
    int b1_forward_projection(double *b, double *x, A_B1 *b1_layer);
    void b1_backward_projection(double *b, double *x, A_B1 *b1_layer);
}