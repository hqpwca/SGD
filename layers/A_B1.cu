#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "A_B1.hh"
#include "../utils/geo.hh"
#include "../utils/matrix_double.hh"
#include "../utils/exception.hh"

#include "spline_conv.hh"

// #define DEBUG
#define Z_SIZE 1

__device__ static int conv_count = 0;

// __device__ static double linear_convolution_1d(double y, double h1, double h2, double h3, double h4, double h5) 
// {
//     atomicAdd(&conv_count, 1);

//     int tot = 0;
//     double hs[5];
//     if(fabs(h1) > 1e-8) hs[tot++] = h1;
//     if(fabs(h2) > 1e-8) hs[tot++] = h2;
//     if(fabs(h3) > 1e-8) hs[tot++] = h3;
//     if(fabs(h4) > 1e-8) hs[tot++] = h4;
//     if(fabs(h5) > 1e-8) hs[tot++] = h5;

//     //printf("%d: %lf, %lf, %lf, %lf, %lf\n", tot, hs[0], hs[1], hs[2], hs[3], hs[4]);

//     switch(tot) {
//         case 3:
//             return forward_difference_3x_1(y, hs[0], hs[1], hs[2]) / 2.0;
//             break;
//         case 5:
//             return forward_difference_5x_1(y, hs[0], hs[1], hs[2], hs[3], hs[4]) / 24.0;
//             break;
//         default:
//             assert(0);
//             break;
//     }
// }

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
__global__ void b1_forward_projection(double *proj, const double *vol, const float *vecs, const float *pm, int3 n3xyz, float3 d3xyz, int nu, float3 src, float3 puv, float3 dtv, int z_size, int np, int ip) {
    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    // int blocky = (blockIdx.y * blockDim.y) + threadIdx.y;
    int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

    // int y_start = blocky * z_size;
    // int y_end = y_start + z_size;
    // y_end = min(y_end, n3xyz.y);

    if(ix >= n3xyz.x) return;
    // if(y_end <= y_start) return;

    int nx = n3xyz.x, ny = n3xyz.y;
    float dx = d3xyz.x, dy = d3xyz.y;
    int maxu, minu;
    int idx, idxu;
    float ox, oy;
    float px = src.x, py = src.y;
    float sx, sy;
    // float u1, u2, u3, u4;
    float us[4] = {0.0};
    float signx1, signx2, signy1, signy2;
    float ax, ay, bx, by, ax1, bx1, ay1, by1, ra, rb, x1, x2, sgn;

    float eps = 1e-7f;
    float C;

    float vx, vy;
    // double a0s, sb0, lsp;
    // double vblur, y, r1;
    float conv;
    double val;

    ox = (ix-0.5f * (nx-1)) * dx;
    // oy = (y_start-0.5f * (ny-1)) * dy;
    oy = (iy-0.5f * (ny-1)) * dy;

    signx1 = ix - 1.0f;
    signx2 = ix + 1.0f;

    bool singular = fabs(puv.x - puv.y) < eps;

    // u1 = pm[0]*signx1 + pm[3];
    // u2 = pm[8]*signx1 + pm[11];

    // u3 = pm[0]*signx2 + pm[3];
    // u4 = pm[8]*signx2 + pm[11];

    // printf("P:(%f, %f)\n", px, py);

    // for(int iy = y_start; iy < y_end; ++ iy, oy += dy) {
        idx = iy*nx+ix;
        C = vol[idx];

        signy1 = iy - 1.0f;
        signy2 = iy + 1.0f;

        if (!singular) {
            us[0] = (pm[0]*signx1 + pm[3] + pm[1]*signy1) / (pm[8]*signx1 + pm[11] + pm[9]*signy1);
            us[1] = (pm[0]*signx1 + pm[3] + pm[1]*signy2) / (pm[8]*signx1 + pm[11] + pm[9]*signy2);
            us[2] = (pm[0]*signx2 + pm[3] + pm[1]*signy1) / (pm[8]*signx2 + pm[11] + pm[9]*signy1);
            us[3] = (pm[0]*signx2 + pm[3] + pm[1]*signy2) / (pm[8]*signx2 + pm[11] + pm[9]*signy2);
        }
        else {
            us[0] = ((pm[0]*signx1 + pm[3] + pm[1]*signy1)/(pm[8]*signx1 + pm[11] + pm[9]*signy1)*1.5f - 1.0f) / puv.x;
            us[1] = ((pm[0]*signx1 + pm[3] + pm[1]*signy2)/(pm[8]*signx1 + pm[11] + pm[9]*signy2)*1.5f - 1.0f) / puv.x;
            us[2] = ((pm[0]*signx2 + pm[3] + pm[1]*signy1)/(pm[8]*signx2 + pm[11] + pm[9]*signy1)*1.5f - 1.0f) / puv.x;
            us[3] = ((pm[0]*signx2 + pm[3] + pm[1]*signy2)/(pm[8]*signx2 + pm[11] + pm[9]*signy2)*1.5f - 1.0f) / puv.x;
        }

        // if (!singular) {
        //     us[0] = (u1 + pm[1]*signy1) / (u2 + pm[9]*signy1);
        //     us[1] = (u1 + pm[1]*signy2) / (u2 + pm[9]*signy2);
        //     us[2] = (u3 + pm[1]*signy1) / (u4 + pm[9]*signy1);
        //     us[3] = (u3 + pm[1]*signy2) / (u4 + pm[9]*signy2);
        // }
        // else {
        //     us[0] = ((u1 + pm[1]*signy1)/(u2 + pm[9]*signy1)*1.5 - 1) / puv.x;
        //     us[1] = ((u1 + pm[1]*signy2)/(u2 + pm[9]*signy2)*1.5 - 1) / puv.x;
        //     us[2] = ((u3 + pm[1]*signy1)/(u4 + pm[9]*signy1)*1.5 - 1) / puv.x;
        //     us[3] = ((u3 + pm[1]*signy2)/(u4 + pm[9]*signy2)*1.5 - 1) / puv.x;
        // }

        // printf("BL: (%f, %f, %f) -> (%f, %f, %f), %f\n", signx1, signy1, 0.0, (u1 + pm[1]*signy1), (pm[4]*signx1+pm[5]*signy1+pm[7]), (u2 + pm[9]*signy1), utest);

        sort4<float>(us, us+1, us+2, us+3);

        minu = min(max(0, (int)floorf(us[0])), nu-1);
        maxu = min(max(0, (int)floorf(us[3])), nu-1);

        sx = dtv.x + puv.x * (minu - 0.5f * nu + 0.5f);
        sy = dtv.y + puv.y * (minu - 0.5f * nu + 0.5f);

        // printf("\tidx: %d, O:(%d(%f), %d(%f)) = %f, Us: [%f, %f, %f, %f], Ur: [%d~%d]\n", idx, ix, ox, iy, oy, vol[idx], us[0], us[1], us[2], us[3], int(minu), int(maxu));

        val = 0;

        for (int ti = 0; ti < maxu - minu + 1; ++ ti, sx += puv.x, sy += puv.y) {
            idxu = (minu + ti) * np + ip;

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

            // printf("\t\t[U=%d] S:(%f, %f), P:(%f, %f), 2Tri(%.12f, %.12f, %.12f, %.12f) = [conv] %.12f\n", minu + ti, sx, sy, px, py, max(vx, vy), min(vx, vy), x1*sgn, x2, conv);

            val = conv * C * dx * dy;
            
            if(idxu < np * nu && idxu >= 0 && val == val && conv > eps)
                atomicAdd(proj+idxu, val);
                // proj[idxu] += val;
        }
    // }
}


__global__ void b1_backward_projection(const double *proj, double *vol, const float *vecs, const float *pm, int3 n3xyz, float3 d3xyz, int nu, float3 src, float3 puv, float3 dtv, int z_size) {
    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    int blocky = (blockIdx.y * blockDim.y) + threadIdx.y;
    // int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

    int y_start = blocky * z_size;
    int y_end = y_start + z_size;
    y_end = min(y_end, n3xyz.y);

    if(ix >= n3xyz.x) return;
    if(y_end <= y_start) return;

    int nx = n3xyz.x, ny = n3xyz.y;
    float dx = d3xyz.x, dy = d3xyz.y;
    int maxu, minu;
    int idx, idxu;
    float ox, oy;
    float px = src.x, py = src.y;
    float sx, sy;
    // float u1, u2, u3, u4;
    float us[4] = {0.0};
    float signx1, signx2, signy1, signy2;
    float ax, ay, bx, by, ax1, bx1, ay1, by1, ra, rb, x1, x2, sgn;

    float eps = 1e-7f;

    float vx, vy;
    // double a0s, sb0, lsp;
    // double vblur, y, r1;
    float conv;
    double val;

    ox = (ix-0.5f * (nx-1)) * dx;
    oy = (y_start-0.5f * (ny-1)) * dy;
    // oy = (iy-0.5f * (ny-1)) * dy;

    signx1 = ix - 1.0f;
    signx2 = ix + 1.0f;

    bool singular = fabs(puv.x - puv.y) < eps;

    // u1 = pm[0]*signx1 + pm[3];
    // u2 = pm[8]*signx1 + pm[11];

    // u3 = pm[0]*signx2 + pm[3];
    // u4 = pm[8]*signx2 + pm[11];

    // printf("P:(%f, %f)\n", px, py);

    for(int iy = y_start; iy < y_end; ++ iy, oy += dy) {
        idx = iy*nx+ix;

        signy1 = iy - 1.0f;
        signy2 = iy + 1.0f;

        if (!singular) {
            us[0] = (pm[0]*signx1 + pm[3] + pm[1]*signy1) / (pm[8]*signx1 + pm[11] + pm[9]*signy1);
            us[1] = (pm[0]*signx1 + pm[3] + pm[1]*signy2) / (pm[8]*signx1 + pm[11] + pm[9]*signy2);
            us[2] = (pm[0]*signx2 + pm[3] + pm[1]*signy1) / (pm[8]*signx2 + pm[11] + pm[9]*signy1);
            us[3] = (pm[0]*signx2 + pm[3] + pm[1]*signy2) / (pm[8]*signx2 + pm[11] + pm[9]*signy2);
        }
        else {
            us[0] = ((pm[0]*signx1 + pm[3] + pm[1]*signy1)/(pm[8]*signx1 + pm[11] + pm[9]*signy1)*1.5f - 1.0f) / puv.x;
            us[1] = ((pm[0]*signx1 + pm[3] + pm[1]*signy2)/(pm[8]*signx1 + pm[11] + pm[9]*signy2)*1.5f - 1.0f) / puv.x;
            us[2] = ((pm[0]*signx2 + pm[3] + pm[1]*signy1)/(pm[8]*signx2 + pm[11] + pm[9]*signy1)*1.5f - 1.0f) / puv.x;
            us[3] = ((pm[0]*signx2 + pm[3] + pm[1]*signy2)/(pm[8]*signx2 + pm[11] + pm[9]*signy2)*1.5f - 1.0f) / puv.x;
        }

        // if (!singular) {
        //     us[0] = (u1 + pm[1]*signy1) / (u2 + pm[9]*signy1);
        //     us[1] = (u1 + pm[1]*signy2) / (u2 + pm[9]*signy2);
        //     us[2] = (u3 + pm[1]*signy1) / (u4 + pm[9]*signy1);
        //     us[3] = (u3 + pm[1]*signy2) / (u4 + pm[9]*signy2);
        // }
        // else {
        //     us[0] = ((u1 + pm[1]*signy1)/(u2 + pm[9]*signy1)*1.5 - 1) / puv.x;
        //     us[1] = ((u1 + pm[1]*signy2)/(u2 + pm[9]*signy2)*1.5 - 1) / puv.x;
        //     us[2] = ((u3 + pm[1]*signy1)/(u4 + pm[9]*signy1)*1.5 - 1) / puv.x;
        //     us[3] = ((u3 + pm[1]*signy2)/(u4 + pm[9]*signy2)*1.5 - 1) / puv.x;
        // }

        // printf("BL: (%f, %f, %f) -> (%f, %f, %f), %f\n", signx1, signy1, 0.0, (u1 + pm[1]*signy1), (pm[4]*signx1+pm[5]*signy1+pm[7]), (u2 + pm[9]*signy1), utest);

        sort4<float>(us, us+1, us+2, us+3);

        minu = min(max(0, (int)floorf(us[0])), nu-1);
        maxu = min(max(0, (int)floorf(us[3])), nu-1);

        sx = dtv.x + puv.x * (minu - 0.5f * nu + 0.5f);
        sy = dtv.y + puv.y * (minu - 0.5f * nu + 0.5f);

        // printf("\tidx: %d, O:(%d(%f), %d(%f)) = %f, Us: [%f, %f, %f, %f], Ur: [%d~%d]\n", idx, ix, ox, iy, oy, vol[idx], us[0], us[1], us[2], us[3], int(minu), int(maxu));

        val = 0;

        for (int ti = 0; ti < maxu - minu + 1; ++ ti, sx += puv.x, sy += puv.y) {
            idxu = minu + ti;

            vx  = vecs[       idxu];
            vy  = vecs[nu   + idxu];

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

            // printf("\t\t[U=%d] S:(%f, %f), P:(%f, %f), vx: %lf, vy: %lf, a0s: %lf, sb0: %lf, lsp: %lf, r1: %lf, vblur: %lf, y: %lf -> %lf [conv] %lf\n", idxu, sx, sy, px, py, vx, vy, a0s, sb0, lsp, r1, vblur, y-a0s*r1, y, conv);

            if(idxu < nu && idxu >= 0 && conv == conv && conv > eps)
                val += conv * proj[idxu] * dx * dy;
        }

        atomicAdd(vol+idx, val);
    }
}

__global__ void bilinear_matrix_generate(double *mat, int3 n3xyz, double3 d3xyz, int nu, double3 src, double3 puv, double3 dtv) { // nz = 1, dz = 0, src.z = 0
    //int ip = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iu = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iu >= nu) return;

    int nx, ny;
    int row;
    double nx2, ny2;
    double dx, dy;
    int i, j;
    double px, py;
    double sx, sy, ax, ay, bx, by;
    double ox, oy;
    double theta, sint, cost;
    double vx, vy, vblur, y;

    nx = n3xyz.x, ny = n3xyz.y;
    nx2 = 0.5 * (nx-1), ny2 = 0.5 * (ny-1);
    dx = d3xyz.x, dy = d3xyz.y;
    px = src.x, py = src.y;
    
    sx = dtv.x + puv.x * (iu - 0.5 * nu + 0.5);
    sy = dtv.y + puv.y * (iu - 0.5 * nu + 0.5);
    ax = sx - puv.x * 0.5;
    ay = sy - puv.y * 0.5;
    bx = sx + puv.x * 0.5;
    by = sy + puv.y * 0.5;

    theta = atan2(fabs(px-sx), fabs(py-sy));
    sint = sin(theta), cost = cos(theta);
    vx = dx * cost, vy = dy * sint;
    
    // TODO: optimize the range of j
    for(i = 0; i < nx; ++i) {
        for(j = 0; j < ny; ++j) {
            ox = (i-nx2) * dx;
            oy = (j-ny2) * dy;
            row = i*ny + j;

            double ra = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((ax-px)*(sx-px) + (ay-py)*(sy-py));
            double rb = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((bx-px)*(sx-px) + (by-py)*(sy-py));
            double ax1 = px + (ax-px) * ra, ay1 = py + (ay-py) * ra;
            double bx1 = px + (bx-px) * rb, by1 = py + (by-py) * rb;
            //y = fabs((sy-py)*ox - (sx-px)*oy + sx*py - sy*px) / sqrtf64((sx-px)*(sx-px) + (sy-py)*(sy-py));

#ifdef DEBUG
            printf("[Slot %d %d %d] ra: %lf, rb: %lf, s:(%lf %lf), p:(%lf, %lf), o:(%lf %lf), a:(%lf %lf), b:(%lf %lf), a1:(%lf %lf), b1:(%lf %lf)\n", iu, i, j, ra, rb, sx, sy, px, py, ox, oy, ax, ay, bx, by, ax1, ay1, bx1, by1);
#endif

            if (fabs(ax1-ox) > dx && fabs(bx1-ox) > dx && fabs(ay1-oy) > dy && fabs(by1-oy) > dy)
                continue;

            vblur = sqrt((bx1-ax1) * (bx1-ax1) + (by1-ay1) * (by1-ay1));
            y = sqrt((ax1-ox) * (ax1-ox) + (ay1-oy) * (ay1-oy));

            if (y > 2 * fabs(vx) + 2 * fabs(vy) + fabs(vblur))
                continue;

            if((ay-py) * (ox-px) > (oy-py) * (ax-px)) 
                y = -y;

            double conv = linear_convolution_1d(y, y+vblur, max(vx, vy), min(vx, vy));

#ifdef DEBUG
            printf("[Conv %d %d %d] theta: %.12lf, vx: %.12lf, vy: %.12lf, vblur: %.12lf, y: %.12lf, conv: %lf\n", iu, i, j, theta, vx, vy, vblur, y, conv);
#endif

            if(conv > 0)
                mat[iu*nx*ny + row] = conv;
        }
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

A_B1::A_B1(GeoData *geo)
{
    geodata = geo;

    ngrid = 1;

    nblock = (geo->nuv.x + (ngrid-1)) / ngrid;

    vblock = dim3(16, 8);

    int bx = (geo->nxyz.x + vblock.x - 1) / vblock.x;
    int by = (geo->nxyz.y + (vblock.y - 1)) / (vblock.y);
    int byz = (geo->nxyz.y + (vblock.y * Z_SIZE - 1)) / (vblock.y * Z_SIZE);

    vgrid = dim3(bx, by);
    vgrid_z = dim3(bx, byz);

    // vecs = new Matrix(geo->np * geo->nuv.x, 5);
    vecs = new Matrix(geo->np * geo->nuv.x, 2);
    vecs->allocateMemory();

    generate_vectors((*vecs)[0], geo);
    vecs->copyHostToDevice();

    printf("%d %d %d %d\n", vgrid.x, vgrid.y, vblock.x, vblock.y);
}

A_B1::~A_B1()
{
    delete vecs;
}

void A_B1::project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
#endif
        b1_forward_projection<<<vgrid_z, vblock>>>(proj(0), vol(0), (*vecs)(p * geodata->nuv.x * 2), geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->nuv.x,
                                                        make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]), Z_SIZE, geodata->np, p);
    }
}

void A_B1::back_project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
#endif

        b1_backward_projection<<<vgrid, vblock>>>(proj(int(p * geodata->nuv.x)), vol(0), (*vecs)(p * geodata->nuv.x * 2), geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->nuv.x,
                                                        make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]), 1);
    }
}


extern "C" {
    int bilinear_matrix_generation(int np, int nu, int nx, int ny, double dx, double dy, double du, double lsd, double lso, double *spmat, int buffer_size) {
        GeoData *geo = new GeoData(nx, ny, 1, nu, 1, np, dx, dy, 0, du, 0);
        geo->geo_init_example(lsd, lso, 0.0f, PI*2 * (np-1)/np);

        int ngrid = 16;

        int nblock = (geo->nuv.x + (ngrid-1)) / ngrid;

        MatrixD mat(nu, nx*ny);
        mat.allocateMemory();

        int idx = 0;

        for(int p=0; p<np; p++){

#ifdef DEBUG
            std::cout << p << std::endl;
#endif

            std::fill(mat[0], mat[nu*nx*ny], 0.0);
            mat.copyHostToDevice();

            bilinear_matrix_generate<<<ngrid, nblock>>>(mat(0), geo->nxyz, geo->dxyz, geo->nuv.x,
                                                        make_double3(*geo->srcs[p*3], *geo->srcs[p*3+1], *geo->srcs[p*3+2]),
                                                        make_double3(*geo->puvs[p*3], *geo->puvs[p*3+1], *geo->puvs[p*3+2]),
                                                        make_double3(*geo->dtvs[p*3], *geo->dtvs[p*3+1], *geo->dtvs[p*3+2]));
            mat.copyDeviceToHost();

            for(int u=0; u<nu; ++u) {
                for(int i=0; i<nx; ++i) {
                    for(int j=0; j<ny; ++j) {
                        if(*mat[u*nx*ny + i*ny + j] != 0){
                            if(idx == buffer_size)
                                assert(0);

                            spmat[3*idx] = p*nu + u;
                            spmat[3*idx+1] = i*ny + j;
                            spmat[3*idx+2] = *mat[u*nx*ny + i*ny + j];
                            ++idx;
                        }
                    }
                }
            }
        }

        delete geo;

        return idx;
    }

    A_B1 *b1_init(int nx, int ny, int np, int nu, double dx, double dy, double du, double lsd, double lso, double *angles){
        GeoData *geo = new GeoData(nx, ny, 1, nu, 1, np, dx, dy, 1, du, 1);
        geo->geo_init_angles(lsd, lso, angles);
        geo->initialize_projection_matrix();

        A_B1 *b1_layer = new A_B1(geo);

        return b1_layer;
    }

    int b1_forward_projection(double *b, double *x, A_B1 *b1_layer) {
        int nx = b1_layer->geodata->nxyz.x;
        int ny = b1_layer->geodata->nxyz.x;

        int np = b1_layer->geodata->np;
        int nu = b1_layer->geodata->nuv.x;

        int cnt = 0;

        MatrixD vol(nx, ny);
        MatrixD proj(np, nu);

        vol.allocateMemory();
        proj.allocateMemory();

        memset(proj[0], 0, np*nu*sizeof(double));
        for(int ix=0; ix<nx; ++ix)
            for(int iy=0; iy<ny; ++iy)
                *(vol[iy * nx + ix]) = x[ix*ny+iy];

        vol.copyHostToDevice();
        proj.copyHostToDevice();

        cudaMemcpyToSymbol(conv_count, &cnt, sizeof(int));

        b1_layer->project(vol, proj, 1.0);

        cudaMemcpyFromSymbol(&cnt, conv_count, sizeof(int));
        // fprintf(stderr ,"Conv count: %d\n", cnt);

        cudaDeviceSynchronize();

        proj.copyDeviceToHost();

        for(int ip=0; ip<np; ++ip)
            for(int iu=0; iu<nu; ++iu)
                b[ip * nu + iu] = *(proj[iu * np + ip]);

        return cnt;
    }

    void b1_backward_projection(double *b, double *x, A_B1 *b1_layer) {
        int nx = b1_layer->geodata->nxyz.x;
        int ny = b1_layer->geodata->nxyz.x;

        int np = b1_layer->geodata->np;
        int nu = b1_layer->geodata->nuv.x;

        MatrixD vol(nx, ny);
        MatrixD proj(np, nu);

        vol.allocateMemory();
        proj.allocateMemory();

        memset(vol[0], 0, nx*ny*sizeof(double));
        memcpy(proj[0], b, np*nu*sizeof(double));

        vol.copyHostToDevice();
        proj.copyHostToDevice();

        b1_layer->back_project(vol, proj, 1.0);

        // fprintf(stderr, "b, x, %p, %p, %lf", b, x, b[10000]);

        cudaDeviceSynchronize();

        vol.copyDeviceToHost();

        // printf("%p, %p, %p, %lf\n", vol[0], vol(0), x, *(vol[100]));

        for(int ix=0; ix<nx; ++ix)
            for(int iy=0; iy<ny; ++iy)
                x[ix*ny+iy] = *(vol[iy * nx + ix]);
    }
}