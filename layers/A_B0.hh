#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_B0{
    int nblock, ngrid;
public:
    GeoData *geodata;

    A_B0(GeoData *geo);
    ~A_B0();

    void project(Matrix &vol, MatrixD &proj, double weight);
    void back_project(Matrix &vol, MatrixD &proj, double weight);
};