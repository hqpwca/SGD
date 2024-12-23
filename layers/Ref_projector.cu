#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "omp.h" 
#include <cuda_runtime.h>

#include "Ref_projector.hh"
#include "../utils/exception.hh"

#define ZSIZE 1

template <class T>
__device__ static void sort2(T* a, T* b)
{
    if (*a > *b)
    {
        T tmp = *b;
        *b = *a;
        *a = tmp;
    }
}

template <class T>
__device__ static void sort4(T* a, T* b, T* c, T* d)
{

    sort2<T>(a,b);
    sort2<T>(c,d);
    sort2<T>(a,c);
    sort2<T>(b,d);
    sort2<T>(b,c);

    // sort3
    // sort2(b, c);
    // sort2(a, b);
    // sort2(b, c);

}

#ifndef REF_CPU

template <class T>
__device__ static T famin(T* a, int len){
    T tmp = 1e50;
    for(int i=0; i<len; ++i) {
        tmp = fmin(tmp, *(a+i));
    }
    return tmp;
}

template <class T>
__device__ static T famax(T* a, int len){
    T tmp = -1e50;
    for(int i=0; i<len; ++i) {
        tmp = fmax(tmp, *(a+i));
    }
    return tmp;
}

#else

template <class T>
__host__ static T famin(T* a, int len){
    T tmp = 1e50;
    for(int i=0; i<len; ++i) {
        tmp = fmin(tmp, *(a+i));
    }
    return tmp;
}

template <class T>
__host__ static T famax(T* a, int len){
    T tmp = -1e50;
    for(int i=0; i<len; ++i) {
        tmp = fmax(tmp, *(a+i));
    }
    return tmp;
}

#endif

// block_size: (8, 8, 64) ILP on z-axis.
#ifdef REF_CPU
__host__ VectorF operator+(const VectorF &a, const VectorF &b){
    return VectorF(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ VectorF operator-(const VectorF &a, const VectorF &b){
    return VectorF(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ VectorF operator*(const VectorF &a, double f){
    return VectorF(a.x*f, a.y*f, a.z*f);
}

__host__ double lineInBox(const double3 p1, const double3 p2, const double3 c1, const double3 d3xyz) {
    double eps = 1e-13;
    double x, y, z;
    double px[6], py[6], pz[6];
    int p = 0;

    double xmin = c1.x, xmax = c1.x + d3xyz.x;
    double ymin = c1.y, ymax = c1.y + d3xyz.y;
    double zmin = c1.z, zmax = c1.z + d3xyz.z;

    double dirx = p2.x - p1.x, diry = p2.y - p1.y, dirz = p2.z - p1.z;

    y = diry/dirx * (xmin-p1.x) + p1.y;
    z = dirz/dirx * (xmin-p1.x) + p1.z;

    if(y > ymin - eps && z > zmin - eps && y < ymax + eps && z < zmax + eps)
        px[p] = xmin, py[p] = y, pz[p] = z, ++p;

    y = diry/dirx * (xmax-p1.x) + p1.y;
    z = dirz/dirx * (xmax-p1.x) + p1.z;

    if(y > ymin - eps && z > zmin - eps && y < ymax + eps && z < zmax + eps)
        px[p] = xmax, py[p] = y, pz[p] = z, ++p;

    x = dirx/diry * (ymin-p1.y) + p1.x;
    z = dirz/diry * (ymin-p1.y) + p1.z;

    if(x > xmin - eps && z > zmin - eps && x < xmax + eps && z < zmax + eps)
        px[p] = x, py[p] = ymin, pz[p] = z, ++p;

    x = dirx/diry * (ymax-p1.y) + p1.x;
    z = dirz/diry * (ymax-p1.y) + p1.z;

    if(x > xmin - eps && z > zmin - eps && x < xmax + eps && z < zmax + eps)
        px[p] = x, py[p] = ymax, pz[p] = z, ++p;
    
    x = dirx/dirz * (zmin-p1.z) + p1.x;
    y = diry/dirz * (zmin-p1.z) + p1.y;

    if(x > xmin - eps && y > ymin - eps && x < xmax + eps && y < ymax + eps)
        px[p] = x, py[p] = y, pz[p] = zmin, ++p;

    x = dirx/dirz * (zmax-p1.z) + p1.x;
    y = diry/dirz * (zmax-p1.z) + p1.y;

    if(x > xmin - eps && y > ymin - eps && x < xmax + eps && y < ymax + eps)
        px[p] = x, py[p] = y, pz[p] = zmax, ++p;

    if(p == 0) return 0.0f;

    xmax = famax<double>(px, p), xmin = famin<double>(px, p);
    ymax = famax<double>(py, p), ymin = famin<double>(py, p);
    zmax = famax<double>(pz, p), zmin = famin<double>(pz, p);

    return sqrtf((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) + (zmax - zmin) * (zmax - zmin));
}

template <class T>
__host__ void simpson_layer(T result, short mdepth, const double3 &src, const double3 &c1, 
                              const double3 &d3xyz, const VectorF &M, const VectorF &U, const VectorF &V, double eps,
                              double factor, double diff) {
    VectorF new_center[4], newU, newV;
    double t[25], qsum, vsum;
    //printf("%lf, %lf, %lf\n", M.x, M.y, M.z);
    //printf("%lf, %lf, %lf\n", U.x, U.y, U.z);
    //printf("%lf, %lf, %lf\n", V.x, V.y, V.z);

    for(int i=-2; i<=2; ++i) {
        for(int j=-2; j<=2; ++j) {
            int idx = (i+2)*5 + j + 2;

            VectorF pt = M + U*double(i) + V*double(j);
            //printf("(%d, %d): %lf, %lf, %lf\n", i, j, pt.x, pt.y, pt.z);

            t[idx] = lineInBox(src, pt.todouble3(), c1, d3xyz);
        }
    }

    vsum = ((t[0] + t[4] + t[20] + t[24]) + 4 * (t[2] + t[10] + t[14] + t[22]) + 16 * t[12]) * factor / 36.0;
    qsum = t[0]+t[4]+t[20]+t[24] + 2*(t[2]+t[10]+t[14]+t[22]) + 4*(t[1]+t[3]+t[5]+t[9]+t[12]+t[15]+t[19]+t[21]+t[23]) +
           8*(t[7]+t[11]+t[13]+t[17]) + 16*(t[6]+t[8]+t[16]+t[18]);
    qsum = qsum * factor / 144.0;

    // if(fabs(qsum-vsum) > diff) {
    //     printf("ERROR: %d %lf %lf %lf %lf\n", mdepth, diff, fabs(qsum-vsum), qsum, vsum);
    //     for(int te=0; te<25; te++) {
    //         printf("%lf ", t[te]);
    //     }
    //     puts("");

    //     for(int i=-2; i<=2; ++i) {
    //         for(int j=-2; j<=2; ++j) {
    //             VectorF pt = M + U*double(i) + V*double(j);
    //             printf("(%d, %d): %lf, %lf, %lf: [%lf]\n", i, j, pt.x, pt.y, pt.z, lineInBox(src, pt.todouble3(), c1, d3xyz));
    //         }
    //     }

    // }
    
    if(fabs(qsum-vsum) <= eps || mdepth <= 0){
        #pragma omp atomic
        *result += qsum + (qsum - vsum) / 15.0;
        return;
    }

    new_center[0] = M-U-V;
    new_center[1] = M+U-V;
    new_center[2] = M-U+V;
    new_center[3] = M+U+V;
    newU = U*0.5, newV = V*0.5;

    #pragma omp parallel for num_threads(16)
    for(int i=0; i<4; ++i) {
        simpson_layer(result, mdepth - 1, src, c1, d3xyz, new_center[i], newU, newV, eps, factor/4, fabs(qsum-vsum));
    }
}


template <class T>
__host__ void Ref_project_Simpson(
                            T proj, const float *vol, int3 n3xyz, double3 d3xyz, const float *pm, 
                            int nu, int nv, double3 uv, double3 vv, double3 src, double3 dtv, 
                            double rect_rect_factor, int z_size)
{
    int ix = 0;
    int iy = 0;
    int oz = 0;

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

    float signy1, signy2, signx1, signx2, signz1, signz2;
    size_t idx, idx0;

    idx0 = (iy*n3xyz.x) + ix;
    
    float u1, u2, u3, u4, u5, u6, u7, u8;
    float v1, v2, v3, v4;

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

        umin = famin<float>(us, 8);
        umax = famax<float>(us, 8);
        vmin = famin<float>(vs, 8);
        vmax = famax<float>(vs, 8);

        min_u = ceilf(umin - 0.5f);
        max_u = ceilf(umax - 0.5f);
        min_v = ceilf(vmin - 0.5f);
        max_v = ceilf(vmax - 0.5f);

        printf("%d %d %d: %f %f %f %f\n", ix, iy, iz, min_u, max_u, min_v, max_v);
        
        if ( ( max_u < 0 ) || ( min_u >= nu ) ) continue;
        if ( ( max_v < 0 ) || ( min_v >= nv ) ) continue;

        for (int ti = 0; ti < max_u - min_u + 1; ++ti) {
            int i = ti + min_u;
            int maxtj = max_v - min_v + 1;

            if (i < 0 || i >= nu) continue;

            // printf("%d\n", i);

            for (int tj = 0; tj < min(maxtj, nv); ++tj) {
                int j = tj + min_v;

                // printf("%d %d\n", i, j);

                double px = dtv.x - (nu*uv.x)/2 - (nv*vv.x)/2 + i*uv.x + j*vv.x;
                double py = dtv.y - (nu*uv.y)/2 - (nv*vv.y)/2 + i*uv.y + j*vv.y;
                double pz = dtv.z - (nu*uv.z)/2 - (nv*vv.z)/2 + i*uv.z + j*vv.z;

                double rel_size = 1.0f;
                double3 du = make_double3(uv.x * rel_size, uv.y * rel_size, uv.z * rel_size);
                double3 dv = make_double3(vv.x * rel_size, vv.y * rel_size, vv.z * rel_size);

                double3 grid[9];
                grid[0] = make_double3(px-du.x/2-dv.x/2, py-du.y/2-dv.y/2, pz-du.z/2-dv.z/2);
                grid[1] = make_double3(px-dv.x/2, py-dv.y/2, pz-dv.z/2);
                grid[2] = make_double3(px+du.x/2-dv.x/2, py+du.y/2-dv.y/2, pz+du.z/2-dv.z/2);
                grid[3] = make_double3(px-du.x/2, py-du.y/2, pz-du.z/2);
                grid[4] = make_double3(px, py, pz);
                grid[5] = make_double3(px+du.x/2, py+du.y/2, pz+du.z/2);
                grid[6] = make_double3(px-du.x/2+dv.x/2, py-du.y/2+dv.y/2, pz-du.z/2+dv.z/2);
                grid[7] = make_double3(px+dv.x/2, py+dv.y/2, pz+dv.z/2);
                grid[8] = make_double3(px+du.x/2+dv.x/2, py+du.y/2+dv.y/2, pz+du.z/2+dv.z/2);

                double3 c1 = make_double3(-(n3xyz.x*d3xyz.x)/2 + ix*d3xyz.x,
                                        -(n3xyz.y*d3xyz.y)/2 + iy*d3xyz.y,
                                        -(n3xyz.z*d3xyz.z)/2 + iz*d3xyz.z); 

                double vgrid[9];
                for(int vid = 0; vid < 9; vid ++) {
                    vgrid[vid] = lineInBox(src, grid[vid], c1, d3xyz);
                }

                idxv = j * nu + i;

                double vsum = ((vgrid[0] + vgrid[2] + vgrid[6] + vgrid[8]) + 4 * (vgrid[1] + vgrid[3] + vgrid[5] + vgrid[7]) + 16 * vgrid[4]) / 36 * C;
                proj[idxv] += vsum;

                //simpson_layer(proj+idxv, 15, src, c1, d3xyz, VectorF(grid[4]), VectorF(du)*0.25, VectorF(dv)*0.25, 1e-9, C, 1e50);
            }
        }
    }
}

#else

__device__ VectorF operator+(const VectorF &a, const VectorF &b){
    return VectorF(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ VectorF operator-(const VectorF &a, const VectorF &b){
    return VectorF(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ VectorF operator*(const VectorF &a, float f){
    return VectorF(a.x*f, a.y*f, a.z*f);
}

__device__ VectorF operator/(const VectorF &a, float f){
    return VectorF(a.x/f, a.y/f, a.z/f);
}

__device__ double lineInBox(const float3 &p1, const float3 &p2, const float3 &c1, const float3 &d3xyz) {
    double eps = 1e-10;
    double x, y, z;
    double px[6], py[6], pz[6];
    int p = 0;

    double xmin = c1.x, xmax = c1.x + d3xyz.x;
    double ymin = c1.y, ymax = c1.y + d3xyz.y;
    double zmin = c1.z, zmax = c1.z + d3xyz.z;

    double dirx = p2.x - p1.x, diry = p2.y - p1.y, dirz = p2.z - p1.z;

    y = diry/dirx * (xmin-p1.x) + p1.y;
    z = dirz/dirx * (xmin-p1.x) + p1.z;

    if(y > ymin - eps && z > zmin - eps && y < ymax + eps && z < zmax + eps)
        px[p] = xmin, py[p] = y, pz[p] = z, ++p;

    y = diry/dirx * (xmax-p1.x) + p1.y;
    z = dirz/dirx * (xmax-p1.x) + p1.z;

    if(y > ymin - eps && z > zmin - eps && y < ymax + eps && z < zmax + eps)
        px[p] = xmax, py[p] = y, pz[p] = z, ++p;

    x = dirx/diry * (ymin-p1.y) + p1.x;
    z = dirz/diry * (ymin-p1.y) + p1.z;

    if(x > xmin - eps && z > zmin - eps && x < xmax + eps && z < zmax + eps)
        px[p] = x, py[p] = ymin, pz[p] = z, ++p;

    x = dirx/diry * (ymax-p1.y) + p1.x;
    z = dirz/diry * (ymax-p1.y) + p1.z;

    if(x > xmin - eps && z > zmin - eps && x < xmax + eps && z < zmax + eps)
        px[p] = x, py[p] = ymax, pz[p] = z, ++p;
    
    x = dirx/dirz * (zmin-p1.z) + p1.x;
    y = diry/dirz * (zmin-p1.z) + p1.y;

    if(x > xmin - eps && y > ymin - eps && x < xmax + eps && y < ymax + eps)
        px[p] = x, py[p] = y, pz[p] = zmin, ++p;

    x = dirx/dirz * (zmax-p1.z) + p1.x;
    y = diry/dirz * (zmax-p1.z) + p1.y;

    if(x > xmin - eps && y > ymin - eps && x < xmax + eps && y < ymax + eps)
        px[p] = x, py[p] = y, pz[p] = zmax, ++p;

    if(p == 0) return 0.0f;

    xmax = famax<double>(px, p), xmin = famin<double>(px, p);
    ymax = famax<double>(py, p), ymin = famin<double>(py, p);
    zmax = famax<double>(pz, p), zmin = famin<double>(pz, p);

    return sqrtf((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) + (zmax - zmin) * (zmax - zmin));
}

template <class T>
__global__ void Ref_project(T proj, const float *vol, int3 n3xyz, float3 d3xyz, const float *pm, 
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
    
    float u1, u2, u3, u4, u5, u6, u7, u8;
    float v1, v2, v3, v4;

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

        umin = famin<float>(us, 8);
        umax = famax<float>(us, 8);
        vmin = famin<float>(vs, 8);
        vmax = famax<float>(vs, 8);

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

                float f2 = lineInBox(src, pxyz, c1, d3xyz) * C;

                idxv = j * nu + i;
                
                if(idxv < nuv && idxv >= 0 && f2 == f2) {
                    atomicAdd(proj+idxv, f2);
                }
            }
        }
    }
}

template <class T>
__device__ void simpson_layer(T result, short mdepth, const float3 &src, const float3 &c1, 
                              const float3 &d3xyz, const VectorF &M, const VectorF &U, const VectorF &V, double eps,
                              double vt[9], double factor) {
    VectorF new_center[4], newU, newV;
    double t[25], qs1, qs2, qs3, qs4, qsum, vsum;
    double nvt[4][9];

    for(int i=-2; i<=2; ++i) {
        for(int j=-2; j<=2; ++j) {
            int idx = (i+2)*5 + j + 2;

            VectorF pt = M + U*(i/4.0f) + V*(j/4.0f);

            if ((i&1) || (j&1)) {
                t[idx] = lineInBox(src, pt.tofloat3(), c1, d3xyz);
            }
            else {
                int vidx = (i+2)/2 * 3 + (j+2)/2;
                t[idx] = vt[vidx];
            }
        }
    }

    qs1 = t[0]+t[2]+t[10]+t[12] + 4*(t[1]+t[5]+t[7]+t[11]) + 16*t[6];
    qs2 = t[2]+t[4]+t[12]+t[14] + 4*(t[3]+t[7]+t[9]+t[13]) + 16*t[8];
    qs3 = t[10]+t[12]+t[20]+t[22] + 4*(t[11]+t[15]+t[17]+t[21]) + 16*t[16];
    qs4 = t[12]+t[14]+t[22]+t[24] + 4*(t[13]+t[17]+t[19]+t[23]) + 16*t[18];

    for(int j=0; j<3; ++j) {
        nvt[0][j] = t[j];
        nvt[1][j] = t[2+j];
        nvt[2][j] = t[10+j];
        nvt[3][j] = t[12+j];
    }
    for(int j=3; j<6; ++j) {
        nvt[0][j] = t[2+j];
        nvt[1][j] = t[4+j];
        nvt[2][j] = t[12+j];
        nvt[3][j] = t[14+j];
    }
    for(int j=6; j<9; ++j) {
        nvt[0][j] = t[4+j];
        nvt[1][j] = t[6+j];
        nvt[2][j] = t[14+j];
        nvt[3][j] = t[16+j];
    }

    vsum = ((vt[0] + vt[2] + vt[6] + vt[8]) + 4 * (vt[1] + vt[3] + vt[5] + vt[7]) + 16 * vt[4]) / 36 * factor;
    qsum = (qs1 + qs2 + qs3 + qs4) / 144 * factor;
    
    if(fabs(qsum-vsum) <= eps || mdepth <= 0){
        atomicAdd(result, qsum + (qsum - vsum) / 15.0);
        return;
    }

    new_center[0] = M-U/4.0f-V/4.0f;
    new_center[1] = M+U/4.0f-V/4.0f;
    new_center[2] = M-U/4.0f+V/4.0f;
    new_center[3] = M+U/4.0f+V/4.0f;
    newU = U/2.0f, newV = V/2.0f;
    
    simpson_layer(result, mdepth - 1, src, c1, d3xyz, new_center[0], newU, newV, eps, nvt[0], factor/2);
    simpson_layer(result, mdepth - 1, src, c1, d3xyz, new_center[1], newU, newV, eps, nvt[1], factor/2);
    simpson_layer(result, mdepth - 1, src, c1, d3xyz, new_center[2], newU, newV, eps, nvt[2], factor/2);
    simpson_layer(result, mdepth - 1, src, c1, d3xyz, new_center[3], newU, newV, eps, nvt[3], factor/2);
}

template <class T>
__global__ void Ref_project_Simpson(
                            T proj, const float *vol, int3 n3xyz, float3 d3xyz, const float *pm, 
                            int nu, int nv, float3 uv, float3 vv, float3 src, float3 dtv, 
                            double rect_rect_factor, int z_size) {
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
    
    float u1, u2, u3, u4, u5, u6, u7, u8;
    float v1, v2, v3, v4;

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

        umin = famin<float>(us, 8);
        umax = famax<float>(us, 8);
        vmin = famin<float>(vs, 8);
        vmax = famax<float>(vs, 8);

        min_u = ceilf(umin - 0.5f);
        max_u = ceilf(umax - 0.5f);
        min_v = ceilf(vmin - 0.5f);
        max_v = ceilf(vmax - 0.5f);
        
        if ( ( max_u < 0 ) || ( min_u >= nu ) ) continue;
        if ( ( max_v < 0 ) || ( min_v >= nv ) ) continue;

        for (int ti = 0; ti < max_u - min_u + 1; ++ti) {
            int i = ti + min_u;

            //printf("%d\n", i);

            for (int tj = 0; tj < max_v - min_v + 1; ++tj) {
                int j = tj + min_v;

                printf("%d %d\n", i, j);

                float px = dtv.x - (nu*uv.x)/2 - (nv*vv.x)/2 + i*uv.x + j*vv.x;
                float py = dtv.y - (nu*uv.y)/2 - (nv*vv.y)/2 + i*uv.y + j*vv.y;
                float pz = dtv.z - (nu*uv.z)/2 - (nv*vv.z)/2 + i*uv.z + j*vv.z;

                float3 du = make_float3(uv.x * 100, uv.y * 100, uv.z * 100);
                float3 dv = make_float3(vv.x * 100, vv.y * 100, vv.z * 100);

                float3 grid[9];
                grid[0] = make_float3(px-du.x/2-dv.x/2, py-du.y/2-dv.y/2, pz-du.z/2-dv.z/2);
                grid[1] = make_float3(px-dv.x/2, py-dv.y/2, pz-dv.z/2);
                grid[2] = make_float3(px+du.x/2-dv.x/2, py+du.y/2-dv.y/2, pz+du.z/2-dv.z/2);
                grid[3] = make_float3(px-du.x/2, py-du.y/2, pz-du.z/2);
                grid[4] = make_float3(px, py, pz);
                grid[5] = make_float3(px+du.x/2, py+du.y/2, pz+du.z/2);
                grid[6] = make_float3(px-du.x/2+dv.x/2, py-du.y/2+dv.y/2, pz-du.z/2+dv.z/2);
                grid[7] = make_float3(px+dv.x/2, py+dv.y/2, pz+dv.z/2);
                grid[8] = make_float3(px+du.x/2+dv.x/2, py+du.y/2+dv.y/2, pz+du.z/2+dv.z/2);

                float3 c1 = make_float3(-(n3xyz.x*d3xyz.x)/2 + ix*d3xyz.x,
                                        -(n3xyz.y*d3xyz.y)/2 + iy*d3xyz.y,
                                        -(n3xyz.z*d3xyz.z)/2 + iz*d3xyz.z); 

                double vgrid[9];
                for(int vid = 0; vid < 9; vid ++) {
                    vgrid[vid] = lineInBox(src, grid[vid], c1, d3xyz);
                }

                idxv = j * nu + i;

                simpson_layer(proj+idxv, 8, src, c1, d3xyz, VectorF(grid[4]), VectorF(du), VectorF(dv), 1e-8, vgrid, C);
            }
        }
    }
}

#endif

// block_size: (8, 8, 64) ILP on z-axis.

void Ref::project(Matrix &vol, Matrix &proj, double weight) { // data processing on device
    for(int p=0; p<geodata->np; p++) {
        //if(p != 0 && p != 15 && p != 35 && p != 45) continue;

        float lsd = *geodata->lsds[p];
        double factor = 1.0;
        std::cout << p << std::endl;

#ifndef REF_CPU
        Ref_project<float *> <<<vgrid, vblock>>>(proj(p * geodata->nuv.x * geodata->nuv.y), vol(0), geodata->nxyz, geodata->dxyz, geodata->pmis(p*12), geodata->nuv.x, geodata->nuv.y,
                                       make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]), 
                                       make_float3(*geodata->pvvs[p*3], *geodata->pvvs[p*3+1], *geodata->pvvs[p*3+2]),
                                       make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                       make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]),
                                       factor, ZSIZE);

        cudaDeviceSynchronize();
        proj.copyDeviceToHost();
#else
        Ref_project_Simpson<float *> (proj[p * geodata->nuv.x * geodata->nuv.y], vol[0], geodata->nxyz, make_double3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->pmis[p*12], geodata->nuv.x, geodata->nuv.y,
                                       make_double3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]), 
                                       make_double3(*geodata->pvvs[p*3], *geodata->pvvs[p*3+1], *geodata->pvvs[p*3+2]),
                                       make_double3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                       make_double3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]),
                                       factor, ZSIZE);
#endif
    }
}

void Ref::project(Matrix &vol, MatrixD &proj, double weight) { // data processing on device
    for(int p=0; p<geodata->np; p++) {
        //if(p != 0 && p != 15 && p != 35 && p != 45) continue;

        float lsd = *geodata->lsds[p];
        double factor = 1.0;
        //std::cout << p << std::endl;

#ifndef REF_CPU
        Ref_project_Simpson<double *> <<<vgrid, vblock>>>(proj(p * geodata->nuv.x * geodata->nuv.y), vol(0), geodata->nxyz, geodata->dxyz, geodata->pmis(p*12), geodata->nuv.x, geodata->nuv.y,
                                       make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]), 
                                       make_float3(*geodata->pvvs[p*3], *geodata->pvvs[p*3+1], *geodata->pvvs[p*3+2]),
                                       make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                       make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]),
                                       factor, ZSIZE);

        cudaDeviceSynchronize();
        proj.copyDeviceToHost();
#else
        Ref_project_Simpson<double *> (proj[p * geodata->nuv.x * geodata->nuv.y], vol[0], geodata->nxyz, make_double3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->pmis[p*12], geodata->nuv.x, geodata->nuv.y,
                                       make_double3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]), 
                                       make_double3(*geodata->pvvs[p*3], *geodata->pvvs[p*3+1], *geodata->pvvs[p*3+2]),
                                       make_double3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                       make_double3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]),
                                       factor, ZSIZE);
#endif
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
    //int bz = (geo->nxyz.z + 63) / 64;
    int bz = geo->nxyz.z;

    vblock = dim3(8, 8, 1);
    vgrid = dim3(bx, by, bz);
}

Ref::~Ref() {
    
}

//Test for only project/backproject kernel;