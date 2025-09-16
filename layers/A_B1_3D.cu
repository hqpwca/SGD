/*
    Using SF projector on z-axis to do 3D forward and backward projection.
    Author: Ke Chen
*/
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "A_B1_3D.hh"
#include "../utils/geo.hh"
#include "../utils/matrix_double.hh"
#include "../utils/exception.hh"

#include "spline_conv.hh"

// #define DEBUG

#ifdef DEBUG
#define Z_SIZE 1
#else
#define Z_SIZE 32
#endif

__device__ static unsigned long long conv_count = 0;

__device__ static void gamma_calculate(float s1, float s2, float *us, float *gamma) {
    float tmp = 0.0f;
    float b1, b2;

    b1 = fmaxf(s1, us[0]);
    b2 = fminf(s2, us[1]);

    if(b2 > b1){
        tmp = (1/(2*(us[1]-us[0]))) * ((b2 - us[0])*(b2 - us[0]) - (b1 - us[0])*(b1 - us[0]));
    }
    else {
        tmp = 0.0f;
    }
    if(tmp == tmp) gamma[0] += tmp;

    b1 = fmaxf(s1, us[1]);
    b2 = fminf(s2, us[2]);

    if(b2 > b1){
        tmp = b2 - b1;
    }
    else{
        tmp = 0.0f;
    }
    if(tmp == tmp) gamma[0] += tmp;

    b1 = fmaxf(s1, us[2]);
    b2 = fminf(s2, us[3]);

    if(b2 > b1){
        tmp = (1/(2*(us[3]-us[2]))) * ((b1 - us[3])*(b1 - us[3]) - (b2 - us[3])*(b2 - us[3]));
    }
    else{
        tmp = 0.0f;
    }
    if(tmp == tmp) gamma[0] += tmp;
}

__device__ static float linear_convolution_1d(float x1, float x2, float a, float b) 
{
    atomicAdd(&conv_count, 1);

    if (b > 1e-7f) 
        return fast_2tri_1box_spline<float>(a, b, x1, x2) / abs(x2-x1);
    else
        return fast_1tri_1box_spline<float>(a, x1, x2) / abs(x2-x1);
        //return forward_difference_3x_1(x2, a, -a, x2-x1) / 2.0;

    //printf("%d: %lf, %lf, %lf, %lf, %lf\n", tot, hs[0], hs[1], hs[2], hs[3], hs[4]);
}

// Fan Beam only
__global__ void b1_3d_forward_projection(double *proj, const double *vol, const float *vecs, const float *pm, int3 n3xyz, float3 d3xyz, int nu, int nv, float3 src, float3 puv, float3 dtv, int z_size) {
    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    // int blocky = (blockIdx.y * blockDim.y) + threadIdx.y;
    int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
    int bz = (blockIdx.z * blockDim.z) + threadIdx.z;

    int z_start = bz * z_size;
    int z_end = z_start + z_size;
    z_end = min(n3xyz.z, z_end);

    if(ix >= n3xyz.x || iy >= n3xyz.y || z_end <= z_start) {
        return;

        #ifdef DEBUG
            printf("Thread: (%d, %d, %d), Block: (%d, %d, %d), z_start: %d, z_end: %d\n", ix, iy, bz, blockIdx.x, blockIdx.y, blockIdx.z, z_start, z_end);
        #endif
    }
    // if(y_end <= y_start) return;

    int nx = n3xyz.x, ny = n3xyz.y, nz = n3xyz.z;
    float dx = d3xyz.x, dy = d3xyz.y, dz = d3xyz.z;
    int maxu, minu, minv, maxv;
    float ox, oy, oz;
    float px = src.x, py = src.y, pz = src.z;
    float sx, sy;
    // float u1, u2, u3, u4;
    float us[4] = {0.0}, vs[4] = {0.0};
    float signx1, signx2, signy1, signy2, signz1, signz2;
    float ax, ay, bx, by, ax1, bx1, ay1, by1, ra, rb, x1, x2, sgn;

    float eps = 1e-7f;
    float C;

    size_t idx, idx0, idxu;

    idx0 = (iy*n3xyz.x) + ix;

    float u1, u2, u3, u4, u5, u6, u7, u8;
    float v1, v2, v3, v4;
    float vx, vy;
    // double a0s, sb0, lsp;
    // double vblur, y, r1;
    float conv;
    double val;

    ox = (ix-0.5f * (nx-1)) * dx;
    // oy = (y_start-0.5f * (ny-1)) * dy;
    oy = (iy-0.5f * (ny-1)) * dy;
    oz = (z_start-0.5f * (nz-1)) * dz;

    signy1 = iy - 1.0f;
    signy2 = iy + 1.0f;
    signx1 = ix - 1.0f;
    signx2 = ix + 1.0f;

    bool singular = fabs(puv.x - puv.y) < eps;

    u1 = pm[0]*signx1 + pm[1]*signy1 + pm[3];
    u2 = pm[8]*signx1 + pm[9]*signy1 + pm[11];

    u3 = pm[0]*signx2 + pm[1]*signy1 + pm[3];
    u4 = pm[8]*signx2 + pm[9]*signy1 + pm[11];

    u5 = pm[0]*signx1 + pm[1]*signy2 + pm[3];
    u6 = pm[8]*signx1 + pm[9]*signy2 + pm[11];

    u7 = pm[0]*signx2 + pm[1]*signy2 + pm[3];
    u8 = pm[8]*signx2 + pm[9]*signy2 + pm[11];

    v1 = pm[4]*signx1 + pm[5]*iy + pm[7];
    v2 = pm[4]*signx2 + pm[5]*iy + pm[7];
    v3 = pm[8]*signx1 + pm[9]*iy + pm[11];
    v4 = pm[8]*signx2 + pm[9]*iy + pm[11];

#ifdef DEBUG
    printf("P:(%f, %f)\n", px, py);
#endif

    if (!singular) {
        us[0] = (u1) / (u2);
        us[1] = (u3) / (u4);
        us[2] = (u5) / (u6);
        us[3] = (u7) / (u8);
    }
    else {
        us[0] = ((u1)/(u2)*1.5f - 1.0f) / puv.x;
        us[1] = ((u3)/(u4)*1.5f - 1.0f) / puv.x;
        us[2] = ((u5)/(u6)*1.5f - 1.0f) / puv.x;
        us[3] = ((u7)/(u8)*1.5f - 1.0f) / puv.x;
    }

    sort4<float>(us, us+1, us+2, us+3);

    minu = min(max(0, (int)floorf(us[0])), nu-1);
    maxu = min(max(0, (int)floorf(us[3])), nu-1);

    if (minu > maxu) return;

    for(int iz = z_start; iz < z_end; ++ iz, oz += dz) {
        idx = ( (size_t) (iz) )*( (size_t) n3xyz.x*n3xyz.y ) + idx0;
        C = vol[idx] * dx * dy * dz;

        signz1 = (iz-0.5f);
        signz2 = (iz+0.5f);

        if (!singular) {
            vs[0] = ( v1 + pm[6] * signz1 ) / ( v3 + pm[10] * signz1 );
            vs[1] = ( v1 + pm[6] * signz2 ) / ( v3 + pm[10] * signz2 );
            vs[2] = ( v2 + pm[6] * signz1 ) / ( v4 + pm[10] * signz1 );
            vs[3] = ( v2 + pm[6] * signz2 ) / ( v4 + pm[10] * signz2 );
        }
        else {
            vs[0] = ((v1 + pm[6] * signz1) / (v3 + pm[10] * signz1) * 1.5f - 1.0f) / puv.y;
            vs[1] = ((v1 + pm[6] * signz2) / (v3 + pm[10] * signz2) * 1.5f - 1.0f) / puv.y;
            vs[2] = ((v2 + pm[6] * signz1) / (v4 + pm[10] * signz1) * 1.5f - 1.0f) / puv.y;
            vs[3] = ((v2 + pm[6] * signz2) / (v4 + pm[10] * signz2) * 1.5f - 1.0f) / puv.y;
        }

        sort4<float>(vs, vs+1, vs+2, vs+3);

        if (vs[0] >= nv) return;

        minv = min(max(0, (int)floorf(vs[0])), nv-1);
        maxv = min(max(0, (int)floorf(vs[3])), nv-1);

        if (minv > maxv) continue;

        float weight = sqrtf(((ox - px) * (ox - px) + (oy - py) * (oy - py) + (oz - pz) * (oz - pz)) / ((ox - px) * (ox - px) + (oy - py) * (oy - py)));
        C = vol[idx] * dx * dy * dz * weight * (2 / ( ((vs[3]-vs[0])+(vs[2]-vs[1])) ));

        // With iz

        // if (!singular) {
        //     us[0] = (u1 + pm[2]*iz) / (u2 + pm[10]*iz);
        //     us[1] = (u3 + pm[2]*iz) / (u4 + pm[10]*iz);
        //     us[2] = (u5 + pm[2]*iz) / (u6 + pm[10]*iz);
        //     us[3] = (u7 + pm[2]*iz) / (u8 + pm[10]*iz);
        // }
        // else {
        //     us[0] = ((u1 + pm[2]*iz)/(u2 + pm[10]*iz)*1.5 - 1) / puv.x;
        //     us[1] = ((u3 + pm[2]*iz)/(u4 + pm[10]*iz)*1.5 - 1) / puv.x;
        //     us[2] = ((u5 + pm[2]*iz)/(u6 + pm[10]*iz)*1.5 - 1) / puv.x;
        //     us[3] = ((u7 + pm[2]*iz)/(u8 + pm[10]*iz)*1.5 - 1) / puv.x;
        // }

#ifdef DEBUG
        // printf("BL: (%f, %f, %f) -> (%f, %f, %f), %f\n", signx1, signy1, 0.0, (u1 + pm[1]*signy1), (pm[4]*signx1+pm[5]*signy1+pm[7]), (u2 + pm[9]*signy1), utest);
#endif

#ifdef DEBUG
        printf("\tidx: %lu, O:(%d(%f), %d(%f), %d(%f)) = %f, Us: [%f, %f, %f, %f], Vs: [%f, %f, %f, %f], Ur: [%d~%d]\n", idx, ix, ox, iy, oy, iz, oz, vol[idx], us[0], us[1], us[2], us[3], vs[0], vs[1], vs[2], vs[3], int(minu), int(maxu));
#endif

        val = 0;

        sx = dtv.x + puv.x * (minu - 0.5f * nu + 0.5f);
        sy = dtv.y + puv.y * (minu - 0.5f * nu + 0.5f);

        for (int ti = 0; ti < maxu - minu + 1; ++ ti, sx += puv.x, sy += puv.y) {
            // idxu = (minu + ti) * np + ip;
            int i = minu + ti;

            vx  = vecs[       minu + ti];
            vy  = vecs[nu   + minu + ti];

            // a0s = vecs[nu*2 + idxu];
            // sb0 = vecs[nu*3 + idxu];
            // lsp = vecs[nu*4 + idxu];

            // r1 = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / (lsp*lsp);
            // vblur = (a0s + sb0) * (-r1);
            // y = crossg(sx-px, sy-py, ox-px, oy-py) / lsp;
            // y += a0s * r1;

            // // if (fabs(y) > fabs(vx) + fabs(vy) + fabs(vblur))
            // //     continue;

            // conv = linear_convolution_1d(y, y+vblur, max(vx, vy), min(vx, vy));

            // printf("\t\t[U=%d] S:(%f, %f), P:(%f, %f), a0s: %f, sb0: %f, lsp: %f, r1: %f, 2Tri(%.12f, %.12f, %.12f, %.12f) = [conv] %.12f\n", idxu, sx, sy, px, py, a0s, sb0, lsp, r1, max(vx, vy), min(vx, vy), y, y+vblur, conv);

            ax = sx - puv.x * 0.5f;
            ay = sy - puv.y * 0.5f;
            bx = sx + puv.x * 0.5f;
            by = sy + puv.y * 0.5f;

            ra = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((ax-px)*(sx-px) + (ay-py)*(sy-py));
            rb = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((bx-px)*(sx-px) + (by-py)*(sy-py));
            ax1 = px + (ax-px) * ra, ay1 = py + (ay-py) * ra;
            bx1 = px + (bx-px) * rb, by1 = py + (by-py) * rb;

            x1 = sqrt((ax1-ox) * (ax1-ox) + (ay1-oy) * (ay1-oy));
            x2 = sqrt((bx1-ox) * (bx1-ox) + (by1-oy) * (by1-oy));

            sgn = ((ax1-ox) * (bx1-ox) + (ay1-oy) * (by1-oy) > 0)?1.0f:-1.0f;

            conv = linear_convolution_1d(x1*sgn, x2, max(vx, vy), min(vx, vy));

#ifdef DEBUG
            printf("\t\t[U=%d] S:(%f, %f), P:(%f, %f), 2Tri(%.12f, %.12f, %.12f, %.12f) = [conv] %.12f\n", minu + ti, sx, sy, px, py, max(vx, vy), min(vx, vy), x1*sgn, x2, conv);
#endif

            val = conv * C;

            if (! (val == val)) continue;

            for (int tj = 0; tj < maxv - minv + 1; ++tj){
                // Dealing with z-axis
                int j = minv + tj;

                float gamma = 0.0f;

                gamma_calculate(j, j + 1.0f, vs, &gamma);

                idxu = j * nu + i;

#ifdef DEBUG
                printf("\t\t\t(U, V) = %d, %d, Gamma: %f, conv = %f, weight = %f, proj[%lu] += %f\n", i, j, gamma, conv, weight, idxu, val * gamma);
#endif

                if(idxu < nu * nv && val == val && gamma > eps){
                    atomicAdd(proj+idxu, val * gamma);
                }
            }
            
            // if(idxu < np * nu && idxu >= 0 && val == val && conv > eps)
            //     atomicAdd(proj+idxu, val);
            //     // proj[idxu] += val;
        }
    }
}

__global__ void b1_3d_backward_projection(const double *proj, double *vol, const float *vecs, const float *pm, int3 n3xyz, float3 d3xyz, int nu, int nv, float3 src, float3 puv, float3 dtv, int z_size) {
int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    // int blocky = (blockIdx.y * blockDim.y) + threadIdx.y;
    int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
    int bz = (blockIdx.z * blockDim.z) + threadIdx.z;

    int z_start = bz * z_size;
    int z_end = z_start + z_size;
    z_end = min(n3xyz.z, z_end);

    if(ix >= n3xyz.x || iy >= n3xyz.y || z_end <= z_start) return;
    // if(y_end <= y_start) return;

    int nx = n3xyz.x, ny = n3xyz.y, nz = n3xyz.z;
    float dx = d3xyz.x, dy = d3xyz.y, dz = d3xyz.z;
    int maxu, minu, minv, maxv;
    float ox, oy, oz;
    float px = src.x, py = src.y, pz = src.z;
    float sx, sy;
    // float u1, u2, u3, u4;
    float us[4] = {0.0}, vs[4] = {0.0};
    float signx1, signx2, signy1, signy2, signz1, signz2;
    float ax, ay, bx, by, ax1, bx1, ay1, by1, ra, rb, x1, x2, sgn;

    float eps = 1e-7f;
    float C;

    size_t idx, idx0, idxu;

    idx0 = (iy*n3xyz.x) + ix;

    float u1, u2, u3, u4, u5, u6, u7, u8;
    float v1, v2, v3, v4;
    float vx, vy;
    // double a0s, sb0, lsp;
    // double vblur, y, r1;
    float conv;
    double val;

    ox = (ix-0.5f * (nx-1)) * dx;
    // oy = (y_start-0.5f * (ny-1)) * dy;
    oy = (iy-0.5f * (ny-1)) * dy;
    oz = (z_start-0.5f * (nz-1)) * dz;

    signy1 = iy - 1.0f;
    signy2 = iy + 1.0f;
    signx1 = ix - 1.0f;
    signx2 = ix + 1.0f;

    bool singular = fabs(puv.x - puv.y) < eps;

    u1 = pm[0]*signx1 + pm[1]*signy1 + pm[3];
    u2 = pm[8]*signx1 + pm[9]*signy1 + pm[11];

    u3 = pm[0]*signx2 + pm[1]*signy1 + pm[3];
    u4 = pm[8]*signx2 + pm[9]*signy1 + pm[11];

    u5 = pm[0]*signx1 + pm[1]*signy2 + pm[3];
    u6 = pm[8]*signx1 + pm[9]*signy2 + pm[11];

    u7 = pm[0]*signx2 + pm[1]*signy2 + pm[3];
    u8 = pm[8]*signx2 + pm[9]*signy2 + pm[11];

    v1 = pm[4]*signx1 + pm[5]*iy + pm[7];
    v2 = pm[4]*signx2 + pm[5]*iy + pm[7];
    v3 = pm[8]*signx1 + pm[9]*iy + pm[11];
    v4 = pm[8]*signx2 + pm[9]*iy + pm[11];

#ifdef DEBUG
    printf("P:(%f, %f)\n", px, py);
#endif

    if (!singular) {
        us[0] = (u1) / (u2);
        us[1] = (u3) / (u4);
        us[2] = (u5) / (u6);
        us[3] = (u7) / (u8);
    }
    else {
        us[0] = ((u1)/(u2)*1.5f - 1.0f) / puv.x;
        us[1] = ((u3)/(u4)*1.5f - 1.0f) / puv.x;
        us[2] = ((u5)/(u6)*1.5f - 1.0f) / puv.x;
        us[3] = ((u7)/(u8)*1.5f - 1.0f) / puv.x;
    }

    sort4<float>(us, us+1, us+2, us+3);

    minu = min(max(0, (int)floorf(us[0])), nu-1);
    maxu = min(max(0, (int)floorf(us[3])), nu-1);

    if (minu >= maxu) return;

    for(int iz = z_start; iz < z_end; ++ iz, oz += dz) {
        idx = ( (size_t) (iz) )*( (size_t) n3xyz.x*n3xyz.y ) + idx0;
        C = vol[idx] * dx * dy * dz;

        signz1 = (iz-0.5f);
        signz2 = (iz+0.5f);

        if (!singular) {
            vs[0] = ( v1 + pm[6] * signz1 ) / ( v3 + pm[10] * signz1 );
            vs[1] = ( v1 + pm[6] * signz2 ) / ( v3 + pm[10] * signz2 );
            vs[2] = ( v2 + pm[6] * signz1 ) / ( v4 + pm[10] * signz1 );
            vs[3] = ( v2 + pm[6] * signz2 ) / ( v4 + pm[10] * signz2 );
        }
        else {
            vs[0] = ((v1 + pm[6] * signz1) / (v3 + pm[10] * signz1) * 1.5f - 1.0f) / puv.y;
            vs[1] = ((v1 + pm[6] * signz2) / (v3 + pm[10] * signz2) * 1.5f - 1.0f) / puv.y;
            vs[2] = ((v2 + pm[6] * signz1) / (v4 + pm[10] * signz1) * 1.5f - 1.0f) / puv.y;
            vs[3] = ((v2 + pm[6] * signz2) / (v4 + pm[10] * signz2) * 1.5f - 1.0f) / puv.y;
        }

        sort4<float>(vs, vs+1, vs+2, vs+3);

        if (vs[0] >= nv) return;

        minv = min(max(0, (int)floorf(vs[0])), nv-1);
        maxv = min(max(0, (int)floorf(vs[3])), nv-1);

        if (minv >= maxv) continue;

        float weight = sqrtf(((ox - px) * (ox - px) + (oy - py) * (oy - py) + (oz - pz) * (oz - pz)) / ((ox - px) * (ox - px) + (oy - py) * (oy - py)));
        C = vol[idx] * dx * dy * dz * weight;

        // With iz

        // if (!singular) {
        //     us[0] = (u1 + pm[2]*iz) / (u2 + pm[10]*iz);
        //     us[1] = (u3 + pm[2]*iz) / (u4 + pm[10]*iz);
        //     us[2] = (u5 + pm[2]*iz) / (u6 + pm[10]*iz);
        //     us[3] = (u7 + pm[2]*iz) / (u8 + pm[10]*iz);
        // }
        // else {
        //     us[0] = ((u1 + pm[2]*iz)/(u2 + pm[10]*iz)*1.5 - 1) / puv.x;
        //     us[1] = ((u3 + pm[2]*iz)/(u4 + pm[10]*iz)*1.5 - 1) / puv.x;
        //     us[2] = ((u5 + pm[2]*iz)/(u6 + pm[10]*iz)*1.5 - 1) / puv.x;
        //     us[3] = ((u7 + pm[2]*iz)/(u8 + pm[10]*iz)*1.5 - 1) / puv.x;
        // }

#ifdef DEBUG
        // printf("BL: (%f, %f, %f) -> (%f, %f, %f), %f\n", signx1, signy1, 0.0, (u1 + pm[1]*signy1), (pm[4]*signx1+pm[5]*signy1+pm[7]), (u2 + pm[9]*signy1), utest);
#endif

#ifdef DEBUG
        if(idx == 0) printf("\tidx: %lu, O:(%d(%f), %d(%f), %d(%f)) = %f, Us: [%f, %f, %f, %f], Vs: [%f, %f, %f, %f], Ur: [%d~%d]\n", idx, ix, ox, iy, oy, iz, oz, vol[idx], us[0], us[1], us[2], us[3], vs[0], vs[1], vs[2], vs[3], int(minu), int(maxu));
#endif

        val = 0;

        sx = dtv.x + puv.x * (minu - 0.5f * nu + 0.5f);
        sy = dtv.y + puv.y * (minu - 0.5f * nu + 0.5f);

        for (int ti = 0; ti < maxu - minu + 1; ++ ti, sx += puv.x, sy += puv.y) {
            // idxu = (minu + ti) * np + ip;
            int i = minu + ti;

            vx  = vecs[       minu + ti];
            vy  = vecs[nu   + minu + ti];

            // a0s = vecs[nu*2 + idxu];
            // sb0 = vecs[nu*3 + idxu];
            // lsp = vecs[nu*4 + idxu];

            // r1 = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / (lsp*lsp);
            // vblur = (a0s + sb0) * (-r1);
            // y = crossg(sx-px, sy-py, ox-px, oy-py) / lsp;
            // y += a0s * r1;

            // // if (fabs(y) > fabs(vx) + fabs(vy) + fabs(vblur))
            // //     continue;

            // conv = linear_convolution_1d(y, y+vblur, max(vx, vy), min(vx, vy));

            // printf("\t\t[U=%d] S:(%f, %f), P:(%f, %f), a0s: %f, sb0: %f, lsp: %f, r1: %f, 2Tri(%.12f, %.12f, %.12f, %.12f) = [conv] %.12f\n", idxu, sx, sy, px, py, a0s, sb0, lsp, r1, max(vx, vy), min(vx, vy), y, y+vblur, conv);

            ax = sx - puv.x * 0.5f;
            ay = sy - puv.y * 0.5f;
            bx = sx + puv.x * 0.5f;
            by = sy + puv.y * 0.5f;

            ra = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((ax-px)*(sx-px) + (ay-py)*(sy-py));
            rb = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((bx-px)*(sx-px) + (by-py)*(sy-py));
            ax1 = px + (ax-px) * ra, ay1 = py + (ay-py) * ra;
            bx1 = px + (bx-px) * rb, by1 = py + (by-py) * rb;

            x1 = sqrt((ax1-ox) * (ax1-ox) + (ay1-oy) * (ay1-oy));
            x2 = sqrt((bx1-ox) * (bx1-ox) + (by1-oy) * (by1-oy));

            sgn = ((ax1-ox) * (bx1-ox) + (ay1-oy) * (by1-oy) > 0)?1.0f:-1.0f;

            conv = linear_convolution_1d(x1*sgn, x2, max(vx, vy), min(vx, vy));

#ifdef DEBUG
            if(idx == 0) printf("\t\t[U=%d] S:(%f, %f), P:(%f, %f), 2Tri(%.12f, %.12f, %.12f, %.12f) = [conv] %.12f\n", minu + ti, sx, sy, px, py, max(vx, vy), min(vx, vy), x1*sgn, x2, conv);
#endif

            if (! (conv == conv && conv > eps)) continue;

            for (int tj = 0; tj < maxv - minv + 1; ++tj){
                // Dealing with z-axis
                int j = minv + tj;

                float gamma = 0.0f;

                gamma_calculate(j, j + 1.0f, vs, &gamma);

                idxu = j * nu + i;

                if(idxu < nu * nv && gamma > eps){
                    val += proj[idxu] * gamma * conv;
                }
            }
        }
        val *= C;

        atomicAdd(vol+idx, val);
    }
}

__host__ static void generate_vectors(float *vecs, GeoData *geo) {
    int np = geo->np;
    int nu = geo->nuv.x;
    double dx = geo->dxyz.x;
    double dy = geo->dxyz.y;

    double cx, cy;
    double ux, uy;
    double px, py;
    double sx, sy;
    // double ax, ay, bx, by;
    double theta;
    double vx, vy;
    // double a0s, sb0, lsp;

    for(int p = 0; p < np; p++) {

        cx = *geo->dtvs[p*3], cy = *geo->dtvs[p*3+1];
        ux = *geo->puvs[p*3], uy = *geo->puvs[p*3+1];
        px = *geo->srcs[p*3], py = *geo->srcs[p*3+1];

        for(int u = 0; u < nu; u++) {
            sx = cx + ux * (u - 0.5 * nu + 0.5);
            sy = cy + uy * (u - 0.5 * nu + 0.5);
            // ax = sx - ux * 0.5;
            // ay = sy - uy * 0.5;
            // bx = sx + ux * 0.5;
            // by = sy + uy * 0.5;

            // lsp = sqrt((px-sx)*(px-sx) + (py-sy)*(py-sy));

            // printf("%d, %d, %lf, %lf, %lf, %lf, %lf\n", p, u, px, py, sx, sy, lsp);

            theta = atan2(fabs(px-sx), fabs(py-sy));
            vx = fabs(dx * cos(theta)), vy = fabs(dy * sin(theta));

            // a0s = fabs(cross(sx-px, sy-py, ax-px, ay-py)) / lsp;
            // sb0 = fabs(cross(sx-px, sy-py, bx-px, by-py)) / lsp;

            // a0s /= ((ax-px)*(sx-px) + (ay-py)*(sy-py)) / (lsp*lsp);
            // sb0 /= ((bx-px)*(sx-px) + (by-py)*(sy-py)) / (lsp*lsp);

            vecs[p*nu*2 + nu*0 + u] = vx;
            vecs[p*nu*2 + nu*1 + u] = vy;
            // vecs[p*nu*5 + nu*2 + u] = a0s;
            // vecs[p*nu*5 + nu*3 + u] = sb0;
            // vecs[p*nu*5 + nu*4 + u] = lsp;
        }
    }
}

__host__ static void generate_vectors_cylindrical(float *vecs, GeoData *geo) {
    int np = geo->np;
    int nu = geo->nuv.x;
    double dx = geo->dxyz.x;
    double dy = geo->dxyz.y;

    double cx, cy;
    double ux, uy;
    double px, py;
    double sx, sy;
    // double ax, ay, bx, by;
    double theta;
    double vx, vy;

    for(int p = 0; p < np; p++) {
        cx = *geo->dtvs[p*3], cy = *geo->dtvs[p*3+1];
        ux = *geo->puvs[p*3], uy = *geo->puvs[p*3+1];
        px = *geo->srcs[p*3], py = *geo->srcs[p*3+1];

        for(int u = 0; u < nu; u++) {
            sx = cx + ux * (u - 0.5 * nu + 0.5);
            sy = cy + uy * (u - 0.5 * nu + 0.5);

            theta = atan2(fabs(px-sx), fabs(py-sy));
            vx = fabs(dx * cos(theta)), vy = fabs(dy * sin(theta));

            vecs[p*nu*2 + nu*0 + u] = vx;
            vecs[p*nu*2 + nu*1 + u] = vy;
        }
    }
}

A_B1_3D::A_B1_3D(GeoData *geo)
{
    geodata = geo;

    ngrid = 1;

    nblock = (geo->nuv.x + (ngrid-1)) / ngrid;

#ifdef DEBUG
    vblock = dim3(1, 1, 1);
#else
    vblock = dim3(16, 8, 1);
#endif

    int bx = (geo->nxyz.x + vblock.x - 1) / vblock.x;
    int by = (geo->nxyz.y + (vblock.y - 1)) / (vblock.y);
    int bz = (geo->nxyz.z + (vblock.z * Z_SIZE - 1)) / (vblock.z * Z_SIZE);

    vgrid = dim3(bx, by, bz);

    // vecs = new Matrix(geo->np * geo->nuv.x, 5);
    vecs = new Matrix(geo->np * geo->nuv.x, 2);
    vecs->allocateMemory();

    generate_vectors((*vecs)[0], geo);
    vecs->copyHostToDevice();

    printf("%d %d %d %d %d %d\n", vgrid.x, vgrid.y, vgrid.z, vblock.x, vblock.y, vblock.z);
}

A_B1_3D::~A_B1_3D()
{
    delete vecs;
}

void A_B1_3D::project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
        cudaDeviceSynchronize();
#endif
        b1_3d_forward_projection<<<vgrid, vblock>>>(proj(int(p * geodata->nuv.x * geodata->nuv.y)), vol(0), (*vecs)(p * geodata->nuv.x * 2), geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->nuv.x, geodata->nuv.y,
                                                        make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]), Z_SIZE);
    }
}

void A_B1_3D::back_project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
        cudaDeviceSynchronize();
#endif

        b1_3d_backward_projection<<<vgrid, vblock>>>(proj(int(p * geodata->nuv.x * geodata->nuv.y)), vol(0), (*vecs)(p * geodata->nuv.x * 2), geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->nuv.x, geodata->nuv.y,
                                                        make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]), Z_SIZE);
    }
}


extern "C" {
    A_B1_3D *b1_3d_init(int nx, int ny, int nz, int np, int nu, int nv, double dx, double dy, double dz, double du, double dv, double lsd, double lso, double *angles){
        GeoData *geo = new GeoData(nx, ny, nz, nu, nv, np, dx, dy, dz, du, dv);
        geo->geo_init_angles(lsd, lso, angles);
        geo->initialize_projection_matrix();

        A_B1_3D *b1_layer = new A_B1_3D(geo);

        return b1_layer;
    }

    unsigned long long b1_3d_forward_projection(double *b, double *x, A_B1_3D *b1_layer) {
        int nx = b1_layer->geodata->nxyz.x;
        int ny = b1_layer->geodata->nxyz.y;
        int nz = b1_layer->geodata->nxyz.z;

        int np = b1_layer->geodata->np;
        int nu = b1_layer->geodata->nuv.x;
        int nv = b1_layer->geodata->nuv.y;

        unsigned long long cnt = 0;

        MatrixD vol(nx*ny*nz, 1);
        MatrixD proj(np*nu*nv, 1);

        vol.allocateMemory();
        proj.allocateMemory();

        memset(proj[0], 0, np*nu*nv*sizeof(double));

        // TODO: Maybe need some modification to account for the data arrangement
        memcpy(vol[0], x, nx*ny*nz*sizeof(double));

        vol.copyHostToDevice();
        proj.copyHostToDevice();

        cudaMemcpyToSymbol(conv_count, &cnt, sizeof(unsigned long long));

        b1_layer->project(vol, proj, 1.0);

        cudaMemcpyFromSymbol(&cnt, conv_count, sizeof(unsigned long long));

        cudaDeviceSynchronize();

        proj.copyDeviceToHost();

        memcpy(b, proj[0], np*nu*nv*sizeof(double));

        return cnt;
    }

    void b1_3d_backward_projection(double *b, double *x, A_B1_3D *b1_layer) {
        int nx = b1_layer->geodata->nxyz.x;
        int ny = b1_layer->geodata->nxyz.x;
        int nz = b1_layer->geodata->nxyz.z;

        int np = b1_layer->geodata->np;
        int nu = b1_layer->geodata->nuv.x;
        int nv = b1_layer->geodata->nuv.y;

        MatrixD vol(nx*ny*nz);
        MatrixD proj(np*nu*nv);

        vol.allocateMemory();
        proj.allocateMemory();

        memset(vol[0], 0, nx*ny*nz*sizeof(double));
        memcpy(proj[0], b, np*nu*nv*sizeof(double));

        vol.copyHostToDevice();
        proj.copyHostToDevice();

        b1_layer->back_project(vol, proj, 1.0);

        // fprintf(stderr, "b, x, %p, %p, %lf", b, x, b[10000]);

        cudaDeviceSynchronize();

        vol.copyDeviceToHost();

        // printf("%p, %p, %p, %lf\n", vol[0], vol(0), x, *(vol[100]));

        memcpy(x, vol[0], nx*ny*nz*sizeof(double));

        // for(int ix=0; ix<nx; ++ix)
        //     for(int iy=0; iy<ny; ++iy)
        //         x[ix*ny+iy] = *(vol[iy * nx + ix]);
    }
}