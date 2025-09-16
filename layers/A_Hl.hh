#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_Hl{
    int nblock, ngrid;
    dim3 vblock, vgrid, vgrid_z;
    Matrix *vecs;
public:
    GeoData *geodata;

    A_Hl(GeoData *geo);
    ~A_Hl();

    void project(MatrixD &vol, MatrixD &proj, double weight);
    void back_project(MatrixD &vol, MatrixD &proj, double weight);
};

extern "C" {
    A_Hl *hl_init(int nx, int ny, int np, int nu, double dt, double du, double lsd, double lso, double *angles, double *dz = nullptr, double *drho = nullptr);
    int hl_forward_projection(double *b, double *x, A_Hl *hl_layer);
    void hl_backward_projection(double *b, double *x, A_Hl *hl_layer);
}