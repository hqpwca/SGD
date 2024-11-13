#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_Hl{
    int nblock, ngrid;
public:
    GeoData *geodata;

    A_Hl(GeoData *geo);
    ~A_Hl();

    void project(Matrix &vol, MatrixD &proj, double weight);
    void back_project(Matrix &vol, MatrixD &proj, double weight);
};