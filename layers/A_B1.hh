#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_B1{
    int nblock, ngrid;
public:
    GeoData *geodata;

    A_B1(GeoData *geo);
    ~A_B1();

    void project(MatrixD &vol, MatrixD &proj, double weight);
    void back_project(MatrixD &vol, MatrixD &proj, double weight);
};