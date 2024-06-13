#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "omp.h" 
#include <cuda_runtime.h>

#include "SF_projector.hh"
#include "../utils/exception.hh"

__device__ void sort2(float* a, float* b)
{
    if (*a > *b)
    {
        float tmp = *b;
        *b = *a;
        *a = tmp;
    }
}

__device__ void sort4(float* a, float* b, float* c, float* d)
{

    sort2(a,b);
    sort2(c,d);
    sort2(a,c);
    sort2(b,d);
    sort2(b,c);

    // sort3
    // sort2(b, c);
    // sort2(a, b);
    // sort2(b, c);

}

// block_size: (8, 8, 64) ILP on z-axis.
__global__ void SF_project(float *proj, const float *vol, int3 n3xyz, float3 d3xyz, const float *pm, int nu, int nv, float3 src, float rect_rect_factor, int z_size)
{
    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
    int oz = (blockIdx.z * blockDim.z) + threadIdx.z;

    int z_start = oz * z_size;
    int z_end = z_start + z_size;
    z_end = min(n3xyz.z, z_end);

    int nx,ny,nz;
    float min_u, max_u, min_v, max_v;
    float s1, s2;
    float us[4] = {0.0};
    float vs[2] = {0.0};
    int idxv;
    float C;

    nx = n3xyz.x, ny = n3xyz.y, nz = n3xyz.z;
    nx = nx/2; ny = ny/2; nz = nz/2;

    float weight, signy1, signy2, signx1, signx2, signz1, signz2;
    unsigned int nuv = nu*nv;
    size_t idx, idx0;

    idx0 = (iy*n3xyz.x) + ix;
        
    float pmv2, pmv3;
    float u1, u2, u3, u4, u5, u6, u7, u8;
    pmv2 = pm[4]*ix + pm[5]*iy + pm[7]; //matrix multiplication result without z-axis
    pmv3 = pm[8]*ix + pm[9]*iy + pm[11]; //matrix multiplication result without z-axis, normalization

    signy1 = iy - 0.5f;
    signy2 = iy + 0.5f;
    signx1 = ix - 0.5f;
    signx2 = ix + 0.5f;

    //matrix multiplication result without z-axis for 4 corners and normalization coefficients
    u1 = pm[0]*signx1 + pm[1]*signy1 + pm[3];
    u2 = pm[8]*signx1 + pm[9]*signy1 + pm[11];

    u3 = pm[0]*signx2 + pm[1]*signy1 + pm[3];
    u4 = pm[8]*signx2 + pm[9]*signy1 + pm[11];

    u5 = pm[0]*signx1 + pm[1]*signy2 + pm[3];
    u6 = pm[8]*signx1 + pm[9]*signy2 + pm[11];

    u7 = pm[0]*signx2 + pm[1]*signy2 + pm[3];
    u8 = pm[8]*signx2 + pm[9]*signy2 + pm[11];

    for (int iz = z_start; iz < z_end; ++iz) {
        idx = ( (size_t) (iz) )*( (size_t) n3xyz.x*n3xyz.y ) + idx0;

        signz1 = (iz-0.5f);
        signz2 = (iz+0.5f);

        vs[0] = ( pmv2 + pm[6] *signz1 ) / ( pmv3 + pm[10]*signz1 );

        min_v = ceilf(vs[0] - 0.5f);
        if ( min_v >= nv ) return;

        vs[1] = ( pmv2 + pm[6] *signz2 ) / ( pmv3 + pm[10]*signz2 );
        
        max_v = ceilf(vs[1] - 0.5f);
        if ( max_v < 0 ) continue;

        C = vol[idx];
        if (C == 0) continue;

        weight = rsqrtf( (d3xyz.x*(ix - nx)-src.x)*(d3xyz.x*(ix - nx)-src.x) + (d3xyz.y*(iy - ny)-src.y)*(d3xyz.y*(iy - ny)-src.y) + (d3xyz.z*(iz - nz)-src.z)*(d3xyz.z*(iz - nz)-src.z) );
        weight *= weight;

        us[0] = ( u1 + pm[2]*iz ) / ( u2 + pm[10]*iz );

        us[1] = ( u3 + pm[2]*iz ) / ( u4 + pm[10]*iz );

        us[2] = ( u5 + pm[2]*iz ) / ( u6 + pm[10]*iz );

        us[3] = ( u7 + pm[2]*iz ) / ( u8 + pm[10]*iz );

        sort4(us, us+1, us+2, us+3);

        min_u = ceilf(us[0] - 0.5f);
        max_u = ceilf(us[3] - 0.5f);
        
        if ( ( max_u < 0 ) || ( min_u >= nu ) ) continue;

        C *= weight * rect_rect_factor;
        
        C *= (2 / ( ((us[3]-us[0])+(us[2]-us[1])) )) * (1 / ( vs[1] - vs[0] ) );

        for (int i = min_u; i < max_u; ++i) {
            s1 = i - 0.5f;
            s2 = i + 0.5f;
            float gamma = 0.0f, tmp = 0.0f;
            float b1, b2;

            b1 = fmaxf(s1, us[0]);
            b2 = fminf(s2, us[1]);

            if(b2 > b1){
                float a = b1 - us[0];
                float b = b2 - b1;
                tmp =  b * b + 2 * a * b / (2 * (us[1] - us[0]));
            }
            else {
                tmp = 0.0f;
            }
            if(tmp == tmp) gamma += tmp;

            b1 = fmaxf(s1, us[1]);
            b2 = fminf(s2, us[2]);

            if(b2 > b1){
                tmp = b2 - b1;
            }
            else{
                tmp = 0.0f;
            }
            if(tmp == tmp) gamma += tmp;

            b1 = fmaxf(s1, us[2]);
            b2 = fminf(s2, us[3]);

            if(b2 > b1){
                float b = b2 - b1;
                float c = us[3] - b2;
                tmp = b * b + 2 * b * c / (2 * (us[3] - us[2]));
            }
            else{
                tmp = 0.0f;
            }
            if(tmp == tmp) gamma += tmp;

            for (int j = min_v; j < max_v; ++j) {
                s1 = j + 0.5f;
                s2 = j - 0.5f;

                C *= gamma * fmaxf(fminf(s1,vs[1]) - fmaxf(s2,vs[0]),0);

                idxv = i * nu + j;
                
                if(idxv < nuv && idxv >= 0 && C == C) {
                    atomicAdd(proj+idxv, C);
                }
            }
        }
    }
}

// block_size: (8, 8, 64) ILP on z-axis.
__global__ void SF_backproject(const float *proj, float *vol, int3 n3xyz, float3 d3xyz, const float *pm, int nu, int nv, float3 src, float rect_rect_factor, int z_size)
{
    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
    int oz = (blockIdx.z * blockDim.z) + threadIdx.z;

    int z_start = oz * z_size;
    int z_end = z_start + z_size;
    z_end = min(n3xyz.z, z_end);

    int nx,ny,nz;
    float min_u, max_u, min_v, max_v;
    float s1, s2;
    float us[4] = {0.0};
    float vs[2] = {0.0};
    int idxv;
    float C;

    nx = n3xyz.x, ny = n3xyz.y, nz = n3xyz.z;
    nx = nx/2; ny = ny/2; nz = nz/2;

    float weight, signy1, signy2, signx1, signx2, signz1, signz2;
    unsigned int nuv = nu*nv;
    size_t idx, idx0;

    idx0 = (iy*n3xyz.x) + ix;
        
    float pmv2, pmv3;
    float u1, u2, u3, u4, u5, u6, u7, u8;
    pmv2 = pm[4]*ix + pm[5]*iy + pm[7]; //matrix multiplication result without z-axis
    pmv3 = pm[8]*ix + pm[9]*iy + pm[11]; //matrix multiplication result without z-axis, normalization

    signy1 = iy - 0.5f;
    signy2 = iy + 0.5f;
    signx1 = ix - 0.5f;
    signx2 = ix + 0.5f;

    //matrix multiplication result without z-axis for 4 corners and normalization coefficients
    u1 = pm[0]*signx1 + pm[1]*signy1 + pm[3];
    u2 = pm[8]*signx1 + pm[9]*signy1 + pm[11];

    u3 = pm[0]*signx2 + pm[1]*signy1 + pm[3];
    u4 = pm[8]*signx2 + pm[9]*signy1 + pm[11];

    u5 = pm[0]*signx1 + pm[1]*signy2 + pm[3];
    u6 = pm[8]*signx1 + pm[9]*signy2 + pm[11];

    u7 = pm[0]*signx2 + pm[1]*signy2 + pm[3];
    u8 = pm[8]*signx2 + pm[9]*signy2 + pm[11];

    for (int iz = z_start; iz < z_end; ++iz) {
        idx = ( (size_t) (iz) )*( (size_t) n3xyz.x*n3xyz.y ) + idx0;

        signz1 = (iz-0.5f);
        signz2 = (iz+0.5f);

        vs[0] = ( pmv2 + pm[6] *signz1 ) / ( pmv3 + pm[10]*signz1 );

        min_v = ceilf(vs[0] - 0.5f);
        if ( min_v >= nv ) return;

        vs[1] = ( pmv2 + pm[6] *signz2 ) / ( pmv3 + pm[10]*signz2 );

        max_v = ceilf(vs[1] - 0.5f);
        if ( max_v < 0 ) continue;

        weight = rsqrtf( (d3xyz.x*(ix - nx)-src.x)*(d3xyz.x*(ix - nx)-src.x) + (d3xyz.y*(iy - ny)-src.y)*(d3xyz.y*(iy - ny)-src.y) + (d3xyz.z*(iz - nz)-src.z)*(d3xyz.z*(iz - nz)-src.z) );
        weight *= weight;

        us[0] = ( u1 + pm[2]*iz ) / ( u2 + pm[10]*iz );

        us[1] = ( u3 + pm[2]*iz ) / ( u4 + pm[10]*iz );

        us[2] = ( u5 + pm[2]*iz ) / ( u6 + pm[10]*iz );

        us[3] = ( u7 + pm[2]*iz ) / ( u8 + pm[10]*iz );

        sort4(us, us+1, us+2, us+3);

        min_u = ceilf(us[0] - 0.5f);
        max_u = ceilf(us[3] - 0.5f);
        
        if ( ( max_u < 0 ) || ( min_u >= nu ) ) continue;

        C = weight * rect_rect_factor * (2 / ( ((us[3]-us[0])+(us[2]-us[1])) )) * (1 / ( vs[1] - vs[0] ) );

        float sumV = 0.0f;

        for (int i = min_u; i < max_u; ++i) {
            s1 = i + 0.5f;
            s2 = i - 0.5f;
            float gamma = 0.0f, tmp = 0.0f;
            float b1, b2;

            b1 = fmaxf(s1, us[0]);
            b2 = fminf(s2, us[1]);

            if(b2 > b1){
                float a = b1 - us[0];
                float b = b2 - b1;
                tmp =  b * b + 2 * a * b / (2 * (us[1] - us[0]));
            }
            else {
                tmp = 0.0f;
            }
            if(tmp == tmp) gamma += tmp;

            b1 = fmaxf(s1, us[1]);
            b2 = fminf(s2, us[2]);

            if(b2 > b1){
                tmp = b2 - b1;
            }
            else{
                tmp = 0.0f;
            }
            if(tmp == tmp) gamma += tmp;

            b1 = fmaxf(s1, us[2]);
            b2 = fminf(s2, us[3]);

            if(b2 > b1){
                float b = b2 - b1;
                float c = us[3] - b2;
                tmp = b * b + 2 * b * c / (2 * (us[3] - us[2]));
            }
            else{
                tmp = 0.0f;
            }
            if(tmp == tmp) gamma += tmp;

            for (int j = min_v; j < max_v; ++j) {
                s1 = j + 0.5f;
                s2 = j - 0.5f;

                float f = gamma * fmaxf(fminf(s1,vs[1]) - fmaxf(s2,vs[0]),0);

                idxv = i * nu + j;
                
                if(idxv < nuv && idxv >= 0) {
                    sumV += proj[idxv] * f;
                }
            }
        }

        sumV *= C;

        if (sumV == sumV){
            atomicAdd(&vol[idx], sumV);
        }

    }
}


void SF::project(Matrix &vol, Matrix &proj, double weight) { // data processing on device
    for(int p=0; p<geodata->np; p++) {
        float lsd = *geodata->lsds[p];
        float factor = lsd * lsd * geodata->dxyz.y * geodata->dxyz.z / (geodata->duv.x * geodata->duv.y);
        //std::cout << p <<' ' << factor << std::endl;

        SF_project<<<vgrid, vblock>>>(proj(p * geodata->nuv.x * geodata->nuv.y), vol(0), geodata->nxyz, geodata->dxyz, geodata->pmis(p*12), geodata->nuv.x, geodata->nuv.y,
                                      make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]), factor, Z_SIZE);
        cudaDeviceSynchronize();
    }
}

void SF::back_project(Matrix &vol, Matrix &proj, double weight) {

}

Matrix& SF::forward(cublasHandle_t &cublasH, Matrix &x) {
    Matrix *y = new Matrix(geodata->nuv.x * geodata->nuv.y, geodata->np);
    y->allocateCudaMemory();
    project(x, *y, 1.0f);

    return *y;
}

Matrix& SF::back_prop(cublasHandle_t &cublasH, Matrix &od, float lr) {
    
}

SF::SF(GeoData *geo) {
    geodata = geo;

    int bx = (geo->nxyz.x + 7) / 8;
    int by = (geo->nxyz.y + 7) / 8;
    int bz = (geo->nxyz.z + 63) / 64;

    vblock = dim3(8, 8, 1);
    vgrid = dim3(bx, by, bz);
}

SF::~SF() {
    
}

//Test for only project/backproject kernel;