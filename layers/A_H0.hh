#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_H0{
    int nblock, ngrid;
    dim3 vblock, vgrid, vgrid_z;
    MatrixD *vecs;
public:
    GeoData *geodata;

    A_H0(GeoData *geo);
    ~A_H0();

    void project(MatrixD &vol, MatrixD &proj, double weight);
    void back_project(MatrixD &vol, MatrixD &proj, double weight);
};

extern "C" {
    A_H0 *h0_init(int nx, int ny, int np, int nu, double dt, double du, double lsd, double lso, double *angles, double *dz = nullptr, double *drho = nullptr);
    int h0_forward_projection(double *b, double *x, A_H0 *h0_layer);
    void h0_backward_projection(double *b, double *x, A_H0 *h0_layer);
}