#pragma once

#include <cuda_runtime.h>
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

class A_H0{
    int nblock, ngrid;
public:
    GeoData *geodata;

    A_H0(GeoData *geo);
    ~A_H0();

    void project(Matrix &vol, MatrixD &proj, double weight);
    void back_project(Matrix &vol, MatrixD &proj, double weight);
};