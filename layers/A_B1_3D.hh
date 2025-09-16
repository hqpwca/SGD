#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_B1_3D{
    int nblock, ngrid;
    dim3 vblock, vgrid;
    Matrix *vecs;

public:
    GeoData *geodata;

    A_B1_3D(GeoData *geo);
    ~A_B1_3D();

    void project(MatrixD &vol, MatrixD &proj, double weight);
    void back_project(MatrixD &vol, MatrixD &proj, double weight);
};

extern "C" {
    A_B1_3D *b1_3d_init(int nx, int ny, int nz, int np, int nu, int nv, double dx, double dy, double dz, double du, double dv, double lsd, double lso, double *angles);
    unsigned long long b1_3d_forward_projection(double *b, double *x, A_B1_3D *b1_layer);
    void b1_3d_backward_projection(double *b, double *x, A_B1_3D *b1_layer);
}