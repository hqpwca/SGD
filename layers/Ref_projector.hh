#pragma once

#include <cuda_runtime.h>
#include "layer.hh"
#include "../utils/matrix.hh"
#include "../utils/matrix_double.hh"
#include "../utils/geo.hh"

#define REF_CPU

class VectorF {
public:
    double x, y, z;

#ifdef REF_CPU
    __host__ VectorF() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ VectorF(double X, double Y, double Z) : x(X), y(Y), z(Z) {}
    __host__ VectorF(float3 xyz) : x(xyz.x), y(xyz.y), z(xyz.z) {}
    __host__ VectorF(double3 xyz) : x(xyz.x), y(xyz.y), z(xyz.z) {}

    __host__ inline float3 tofloat3() {
        return make_float3(x, y, z);
    }
    __host__ inline double3 todouble3() {
        return make_double3(x, y, z);
    }
#else
    __device__ VectorF() : x(0.0f), y(0.0f), z(0.0f) {}
    __device__ VectorF(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    __device__ VectorF(float3 xyz) : x(xyz.x), y(xyz.y), z(xyz.z) {}

    __device__ inline float3 tofloat3() {
        return make_float3(x, y, z);
    }
#endif
};

class Ref : public Layer{
    dim3 vblock, vgrid;
public:
    GeoData *geodata;

    Ref(GeoData *geo);
    ~Ref();

    void project(Matrix &vol, Matrix &proj, double weight);
    void project(Matrix &vol, MatrixD &proj, double weight);

    Matrix& forward(cublasHandle_t &cublasH, Matrix &x); // x: (num_input, batch_size)
    Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od, float lr = 0.01); // od: (num_output, batch_size)
};