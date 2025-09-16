#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "matrix.hh"
#include "matrix_double.hh"

#define PI 3.14159265358979323846

class GeoData {
public:
    int3 nxyz;
    int2 nuv;
    double3 dxyz;
    double2 duv;
    int np;

    MatrixD srcs;
    MatrixD puvs;
    MatrixD pvvs;
    MatrixD dtvs;
    MatrixD ucs;
    MatrixD vcs;
    MatrixD pms;
    Matrix pmis;
    MatrixD lsds;
    MatrixD lsos;

    GeoData(int nx, int ny, int nz, int nu, int nv, int np, double dx, double dy, double dz, double du, double dv);
    ~GeoData();

    void initialize_projection_matrix();

    void geo_init_example(double lsd, double lso, double start_angle, double end_angle);
    void geo_init_angles(double lsd, double lso, double *angles, double *dz = nullptr, double *drho = nullptr);

    void geo_init_helical(double lsd, double lso, double dz, double *angles);
};
