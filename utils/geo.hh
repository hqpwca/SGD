#include <cuda_runtime.h>
#include <memory>

#include "matrix.hh"
#include "matrix_double.hh"

#define PI 3.141592653589793f

class GeoData {
public:
    int3 nxyz;
    int2 nuv;
    float3 dxyz;
    float2 duv;
    int np;

    Matrix srcs;
    Matrix puvs;
    Matrix pvvs;
    Matrix dtvs;
    Matrix ucs;
    Matrix vcs;
    Matrix pms;
    Matrix pmis;
    Matrix lsds;
    Matrix lsos;

    GeoData(int nx, int ny, int nz, int nu, int nv, int np, float dx, float dy, float dz, float du, float dv);
    ~GeoData();

    void initialize_projection_matrix();

    void geo_init_example(float lsd, float lso, float start_angle, float end_angle);
};