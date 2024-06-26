#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "omp.h" 
#include <cuda_runtime.h>

#include "Ref_projector.hh"
#include "../utils/exception.hh"

__device__ static void sort2(float* a, float* b)
{
    if (*a > *b)
    {
        float tmp = *b;
        *b = *a;
        *a = tmp;
    }
}

__device__ static void sort4(float* a, float* b, float* c, float* d)
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

__device__ static float famin(float* a, int len){
    float tmp = 1e50;
    for(int i=0; i<len; ++i) {
        tmp = fmin(tmp, *(a+i));
    }
    return tmp;
}

__device__ static float famax(float* a, int len){
    float tmp = -1e50;
    for(int i=0; i<len; ++i) {
        tmp = fmax(tmp, *(a+i));
    }
    return tmp;
}

__device__ static double famin(double* a, int len){
    double tmp = 1e50;
    for(int i=0; i<len; ++i) {
        tmp = fmin(tmp, *(a+i));
    }
    return tmp;
}

__device__ static double famax(double* a, int len){
    double tmp = -1e50;
    for(int i=0; i<len; ++i) {
        tmp = fmax(tmp, *(a+i));
    }
    return tmp;
}

__device__ static void sort2(double* a, double* b)
{
    if (*a > *b)
    {
        double tmp = *b;
        *b = *a;
        *a = tmp;
    }
}

__device__ static void sort4(double* a, double* b, double* c, double* d)
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

__device__ float lineInBox(const float3 *p1, const float3 *p2, const float3 *c1, const float3 *d3xyz) {
    double eps = 1e-10;
    double x, y, z;
    double px[6], py[6], pz[6];
    int p = 0;

    double xmin = c1->x, xmax = c1->x + d3xyz->x;
    double ymin = c1->y, ymax = c1->y + d3xyz->y;
    double zmin = c1->z, zmax = c1->z + d3xyz->z;

    double dirx = p2->x - p1->x, diry = p2->y - p1->y, dirz = p2->z - p1->z;

    y = diry/dirx * (xmin-p1->x) + p1->y;
    z = dirz/dirx * (xmin-p1->x) + p1->z;

    if(y > ymin - eps && z > zmin - eps && y < ymax + eps && z < zmax + eps)
        px[p] = xmin, py[p] = y, pz[p] = z, ++p;

    y = diry/dirx * (xmax-p1->x) + p1->y;
    z = dirz/dirx * (xmax-p1->x) + p1->z;

    if(y > ymin - eps && z > zmin - eps && y < ymax + eps && z < zmax + eps)
        px[p] = xmax, py[p] = y, pz[p] = z, ++p;

    x = dirx/diry * (ymin-p1->y) + p1->x;
    z = dirz/diry * (ymin-p1->y) + p1->z;

    if(x > xmin - eps && z > zmin - eps && x < xmax + eps && z < zmax + eps)
        px[p] = x, py[p] = ymin, pz[p] = z, ++p;

    x = dirx/diry * (ymax-p1->y) + p1->x;
    z = dirz/diry * (ymax-p1->y) + p1->z;

    if(x > xmin - eps && z > zmin - eps && x < xmax + eps && z < zmax + eps)
        px[p] = x, py[p] = ymax, pz[p] = z, ++p;
    
    x = dirx/dirz * (zmin-p1->z) + p1->x;
    y = diry/dirz * (zmin-p1->z) + p1->y;

    if(x > xmin - eps && y > ymin - eps && x < xmax + eps && y < ymax + eps)
        px[p] = x, py[p] = y, pz[p] = zmin, ++p;

    x = dirx/dirz * (zmax-p1->z) + p1->x;
    y = diry/dirz * (zmax-p1->z) + p1->y;

    if(x > xmin - eps && y > ymin - eps && x < xmax + eps && y < ymax + eps)
        px[p] = x, py[p] = y, pz[p] = zmax, ++p;

    if(p == 0) return 0.0f;

    xmax = famax(px, p), xmin = famin(px, p);
    ymax = famax(py, p), ymin = famin(py, p);
    zmax = famax(pz, p), zmin = famin(pz, p);

    return sqrtf((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) + (zmax - zmin) * (zmax - zmin));
}

// block_size: (8, 8, 64) ILP on z-axis.
__global__ void Ref_project(float *proj, const float *vol, int3 n3xyz, float3 d3xyz, const float *pm, 
                            int nu, int nv, float3 uv, float3 vv, float3 src, float3 dtv, 
                            double rect_rect_factor, int z_size)
{
    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
    int oz = (blockIdx.z * blockDim.z) + threadIdx.z;

    int z_start = oz * z_size;
    int z_end = z_start + z_size;
    z_end = min(n3xyz.z, z_end);

    if(ix >= n3xyz.x || iy >= n3xyz.y || z_end <= z_start) return;

    int nx,ny,nz;
    float min_u, max_u, min_v, max_v;
    float umin, umax, vmin, vmax;
    float us[8] = {0.0};
    float vs[8] = {0.0};
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
    float v1, v2, v3, v4;
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

    v1 = pm[4]*signx1 + pm[5]*signy1 + pm[7];
    v2 = pm[4]*signx2 + pm[5]*signy1 + pm[7];
    v3 = pm[4]*signx1 + pm[5]*signy2 + pm[7];
    v4 = pm[4]*signx2 + pm[5]*signy2 + pm[7];

    for (int iz = z_start; iz < z_end; ++iz) {
        idx = ( (size_t) (iz) )*( (size_t) n3xyz.x*n3xyz.y ) + idx0;

        signz1 = (iz-0.5f);
        signz2 = (iz+0.5f);

        C = vol[idx];
        if (C == 0) continue;

        weight = rsqrtf( (d3xyz.x*(ix - nx)-src.x)*(d3xyz.x*(ix - nx)-src.x) + (d3xyz.y*(iy - ny)-src.y)*(d3xyz.y*(iy - ny)-src.y) + (d3xyz.z*(iz - nz)-src.z)*(d3xyz.z*(iz - nz)-src.z) );
        weight *= weight;

        us[0] = ( u1 + pm[2]*signz1 ) / ( u2 + pm[10]*signz1 );
        us[1] = ( u3 + pm[2]*signz1 ) / ( u4 + pm[10]*signz1 );
        us[2] = ( u5 + pm[2]*signz1 ) / ( u6 + pm[10]*signz1 );
        us[3] = ( u7 + pm[2]*signz1 ) / ( u8 + pm[10]*signz1 );
        us[4] = ( u1 + pm[2]*signz2 ) / ( u2 + pm[10]*signz2 );
        us[5] = ( u3 + pm[2]*signz2 ) / ( u4 + pm[10]*signz2 );
        us[6] = ( u5 + pm[2]*signz2 ) / ( u6 + pm[10]*signz2 );
        us[7] = ( u7 + pm[2]*signz2 ) / ( u8 + pm[10]*signz2 );

        vs[0] = ( v1 + pm[6]*signz1 ) / ( u2 + pm[10]*signz1 );
        vs[1] = ( v2 + pm[6]*signz1 ) / ( u4 + pm[10]*signz1 );
        vs[2] = ( v3 + pm[6]*signz1 ) / ( u6 + pm[10]*signz1 );
        vs[3] = ( v4 + pm[6]*signz1 ) / ( u8 + pm[10]*signz1 );
        vs[4] = ( v1 + pm[6]*signz2 ) / ( u2 + pm[10]*signz2 );
        vs[5] = ( v2 + pm[6]*signz2 ) / ( u4 + pm[10]*signz2 );
        vs[6] = ( v3 + pm[6]*signz2 ) / ( u6 + pm[10]*signz2 );
        vs[7] = ( v4 + pm[6]*signz2 ) / ( u8 + pm[10]*signz2 );

        umin = famin(us, 8);
        umax = famax(us, 8);
        vmin = famin(vs, 8);
        vmax = famax(vs, 8);

        min_u = ceilf(umin - 0.5f);
        max_u = ceilf(umax - 0.5f);
        min_v = ceilf(vmin - 0.5f);
        max_v = ceilf(vmax - 0.5f);
        
        if ( ( max_u < 0 ) || ( min_u >= nu ) ) continue;
        if ( ( max_v < 0 ) || ( min_v >= nv ) ) continue;

        for (int ti = 0; ti < max_u - min_u + 1; ++ti) {
            int i = ti + min_u;

            for (int tj = 0; tj < max_v - min_v + 1; ++tj) {
                int j = tj + min_v;

                float px = dtv.x - (nu*uv.x)/2 - (nv*vv.x)/2 + i*uv.x + j*vv.x;
                float py = dtv.y - (nu*uv.y)/2 - (nv*vv.y)/2 + i*uv.y + j*vv.y;
                float pz = dtv.z - (nu*uv.z)/2 - (nv*vv.z)/2 + i*uv.z + j*vv.z;

                float3 pxyz = make_float3(px, py, pz);

                float3 c1 = make_float3(-(n3xyz.x*d3xyz.x)/2 + ix*d3xyz.x,
                                        -(n3xyz.y*d3xyz.y)/2 + iy*d3xyz.y,
                                        -(n3xyz.z*d3xyz.z)/2 + iz*d3xyz.z);

                float f2 = lineInBox(&src, &pxyz, &c1, &d3xyz) * C;

                idxv = j * nu + i;
                
                if(idxv < nuv && idxv >= 0 && f2 == f2) {
                    atomicAdd(proj+idxv, f2);
                }
            }
        }
    }
}

// block_size: (8, 8, 64) ILP on z-axis.

void Ref::project(Matrix &vol, Matrix &proj, double weight) { // data processing on device
    for(int p=0; p<geodata->np; p++) {
        if(p != 0 && p != 15 && p != 35 && p != 45) continue;

        float lsd = *geodata->lsds[p];
        double factor = 1.0;
        //std::cout << p << std::endl;

        Ref_project<<<vgrid, vblock>>>(proj(p * geodata->nuv.x * geodata->nuv.y), vol(0), geodata->nxyz, geodata->dxyz, geodata->pmis(p*12), geodata->nuv.x, geodata->nuv.y,
                                       make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]), 
                                       make_float3(*geodata->pvvs[p*3], *geodata->pvvs[p*3+1], *geodata->pvvs[p*3+2]),
                                       make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                       make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]),
                                       factor, Z_SIZE);
        cudaDeviceSynchronize();
    }
}

Matrix& Ref::forward(cublasHandle_t &cublasH, Matrix &x) {
    Matrix *y = new Matrix(geodata->nuv.x * geodata->nuv.y, geodata->np);
    y->allocateCudaMemory();
    project(x, *y, 1.0f);

    return *y;
}

Matrix& Ref::back_prop(cublasHandle_t &cublasH, Matrix &od, float lr) {
    
}

Ref::Ref(GeoData *geo) {
    geodata = geo;

    int bx = (geo->nxyz.x + 7) / 8;
    int by = (geo->nxyz.y + 7) / 8;
    int bz = (geo->nxyz.z + 63) / 64;

    vblock = dim3(8, 8, 1);
    vgrid = dim3(bx, by, bz);
}

Ref::~Ref() {
    
}

//Test for only project/backproject kernel;