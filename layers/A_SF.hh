#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_SF{
    int nblock, ngrid;
    dim3 vblock, vgrid, vgrid_z;

public:
    GeoData *geodata;

    A_SF(GeoData *geo);
    ~A_SF();

    void project(MatrixD &vol, MatrixD &proj, double weight);
    void back_project(MatrixD &vol, MatrixD &proj, double weight);
};

extern "C" {
    A_SF *SF_init(int nx, int ny, int np, int nu, double dx, double dy, double du, double lsd, double lso, double *angles);
    int SF_forward_projection(double *b, double *x, A_SF *SF_layer);
    void SF_backward_projection(double *b, double *x, A_SF *SF_layer);
}