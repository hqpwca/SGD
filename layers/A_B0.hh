#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_B0{
    int nblock, ngrid;
    dim3 vblock, vgrid;
    MatrixD *vecs;
    
public:
    GeoData *geodata;

    A_B0(GeoData *geo);
    ~A_B0();

    void project(MatrixD &vol, MatrixD &proj, double weight);
    void back_project(MatrixD &vol, MatrixD &proj, double weight);
};

extern "C" {
    A_B0 *b0_init(int nx, int ny, int np, int nu, double dx, double dy, double du, double lsd, double lso, double *angles, double *dz = nullptr, double *drho = nullptr);
    int b0_forward_projection(double *b, double *x, A_B0 *b0_layer);
    void b0_backward_projection(double *b, double *x, A_B0 *b0_layer);
}