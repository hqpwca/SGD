#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>

#include "A_Hl.hh"
#include "../utils/geo.hh"
#include "../utils/matrix_double.hh"
#include "../utils/exception.hh"
#include "spline_conv.hh"

#define SQRT3 1.7320508075688772935274463415

// #define DEBUG
#define Z_SIZE 1

__device__ static int conv_count = 0;

__device__ static double linear_convolution_1d(double y, double h1, double h2, double h3, double h4) 
{
    int tot = 0;
    double hs[4];
    if(fabs(h1) > 1e-8) hs[tot++] = h1;
    if(fabs(h2) > 1e-8) hs[tot++] = h2;
    if(fabs(h3) > 1e-8) hs[tot++] = h3;
    if(fabs(h4) > 1e-8) hs[tot++] = h4;

    //printf("%d: %lf, %lf, %lf, %lf, %lf\n", tot, hs[0], hs[1], hs[2], hs[3], hs[4]);

    switch(tot) {
        case 2:
            return forward_difference_2x_1(y, hs[0], hs[1]);
            break;
        case 3:
            return forward_difference_3x_1(y, hs[0], hs[1], hs[2]) / 2.0;
            break;
        case 4:
            return forward_difference_4x_1(y, hs[0], hs[1], hs[2], hs[3]) / 6.0;
            break;
        default:
            assert(0);
            break;
    }
}

// ---- constants (tweakable) ----
#ifndef PM_W_EPS
#define PM_W_EPS 1e-8f   // threshold for |w| considered singular
#endif
#ifndef JITTER_EPS
#define JITTER_EPS 1e-3f // small index jitter to escape exact singularities
#endif

// pmi rows: u_row(0:3), v_row(4:7), w_row(8:11)
// project u with singular guard; returns false if invalid
__device__ __forceinline__
bool project_u_safe(const float* __restrict__ pmi, float ix, float iy, float& u_out) {
    float num = fmaf(pmi[0], ix, fmaf(pmi[1], iy, pmi[3]));
    float den = fmaf(pmi[8], ix, fmaf(pmi[9], iy, pmi[11]));
    if (!isfinite(num) || !isfinite(den) || fabsf(den) < PM_W_EPS) return false;
    float u = num / den;
    if (!isfinite(u)) return false;
    u_out = u;
    return true;
}

// compute u, dudx, dudy at a point (ix,iy) using quotient rule; returns false if singular
__device__ __forceinline__
bool project_u_and_jacobian(const float* __restrict__ pmi, float ix, float iy,
                            float& u0, float& du_dx, float& du_dy) {
    float Ax = pmi[0], Ay = pmi[1], Ac = pmi[3];
    float Gx = pmi[8], Gy = pmi[9], Gc = pmi[11];

    float num = fmaf(Ax, ix, fmaf(Ay, iy, Ac));
    float den = fmaf(Gx, ix, fmaf(Gy, iy, Gc));
    if (!isfinite(num) || !isfinite(den) || fabsf(den) < PM_W_EPS) return false;

    float inv_den  = 1.0f / den;
    float inv_den2 = inv_den * inv_den;
    u0 = num * inv_den;

    // du/dx = (Ax*den - num*Gx) / den^2
    du_dx = (Ax * den - num * Gx) * inv_den2;
    // du/dy = (Ay*den - num*Gy) / den^2
    du_dy = (Ay * den - num * Gy) * inv_den2;

    return isfinite(u0) && isfinite(du_dx) && isfinite(du_dy);
}

// Robust 6-corner bounds for hex cell at (ic, il) in index space
// Assumes pmi was built with dx=dt*sqrt(3), dy=dt*0.5
__device__ __forceinline__
void hex_u_bounds_6_safe(const float* __restrict__ pmi,
                         int ic, int il, int nu,
                         int& minu, int& maxu)
{
    // center with odd-row 1/2-column shift
    const float ic_c = (float)ic + ((il & 1) ? 0.5f : 0.0f);
    const float il_c = (float)il;

    // 6 vertex offsets (pointy-top): (Δx, Δy) in INDEX units
    const float offc[6] = { 0.0f, +0.5f, +0.5f,  0.0f, -0.5f, -0.5f };
    const float offl[6] = { -2.0f, -1.0f, +1.0f, +2.0f, +1.0f, -1.0f };

    float umin =  1e30f, umax = -1e30f;
    int valid = 0;

    // 1) try exact vertices
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        float u;
        if (project_u_safe(pmi, ic_c + offc[k], il_c + offl[k], u)) {
            umin = fminf(umin, u);
            umax = fmaxf(umax, u);
            // printf("V(%d): (%f, %f) [%f, %f] -> u=%f\n", k, ic_c + offc[k], il_c + offl[k], (ic_c+ offc[k]) * SQRT3, (il_c + offl[k]) * 0.5f,  u);
            ++valid;
        }
    }

    // 2) if too few valid, jitter the vertices slightly to escape exact w=0
    if (valid < 2) {
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            float u;
            float ix = ic_c + offc[k] + ((k & 1) ? +JITTER_EPS : -JITTER_EPS);
            float iy = il_c + offl[k] + ((k & 2) ? +JITTER_EPS : -JITTER_EPS);
            if (project_u_safe(pmi, ix, iy, u)) {
                umin = fminf(umin, u);
                umax = fmaxf(umax, u);
                ++valid;
            }
        }
    }

    // 3) if still underconstrained, linearize u(ix,iy) at the center and estimate corners
    if (valid < 2) {
        float u0, du_dx, du_dy;
        if (project_u_and_jacobian(pmi, ic_c, il_c, u0, du_dx, du_dy)) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                float du = du_dx * offc[k] + du_dy * offl[k];
                float u  = u0 + du;
                if (isfinite(u)) {
                    umin = fminf(umin, u);
                    umax = fmaxf(umax, u);
                    ++valid;
                }
            }
        }
    }

    // 4) last resort: full scan
    if (valid < 2) {
        minu = 0;
        maxu = nu - 1;
        return;
    }

    // Convert to integer detector bins
    // (clamp to [0, nu-1], and ensure at least one bin)
    minu = min(max(0, (int)floorf(umin)), nu - 1);
    maxu = min(max(0, (int)floorf(umax)), nu - 1);
    if (maxu < minu) maxu = minu;
}

__global__ void hl_forward_projection(double *proj, const double *vol, const float *vecs, const float *pm, int3 n3xyz, float3 d3xyz, int nu, float3 src, float3 puv, float3 dtv, int z_size, int np, int ip) {
    int ic = (blockIdx.x * blockDim.x) + threadIdx.x;
    // int blocky = (blockIdx.y * blockDim.y) + threadIdx.y;
    int il = (blockIdx.y * blockDim.y) + threadIdx.y;

    // int y_start = blocky * z_size;
    // int y_end = y_start + z_size;
    // y_end = min(y_end, n3xyz.y);
    // if(y_end <= y_start) return;

    int nc = n3xyz.x, nl = n3xyz.y;

    if (ic >= nc) return;
    if (il >= nl) return;
    if ((il & 1) && ic == (nc - 1)) return;

    float dt = d3xyz.y * 2.0f;
    float dx = dt * (float)SQRT3;
    float dy = dt * 0.5f;

    float nl2 = 0.5f * (nl - 1);
    float nc2 = 0.5f * (nc - 1);
    float nc3 = 0.5f * (nc - 2);

    float oy = (il - nl2) * dy;
    float ox = ((il & 1) ? (ic - nc3) : (ic - nc2)) * dx;

    float px = src.x, py = src.y;

    // printf("P:(%f, %f), O:(%d(%f), %d(%f))\n", px, py, ic, ox, il, oy);

    int idx = il * nc + ic;
    double C = vol[idx];

    float eps = 1e-7f;
    bool singular = fabsf(puv.x - puv.y) < eps;

    // float ic_eff = (float)ic + ((il & 1) ? 0.5f : 0.0f);

    int minu, maxu;
    hex_u_bounds_6_safe(pm, ic, il, nu, minu, maxu);

    float sx = dtv.x + puv.x * (minu - 0.5f * nu + 0.5f);
    float sy = dtv.y + puv.y * (minu - 0.5f * nu + 0.5f);

    (void)vecs;

    // printf("BL: (%f, %f, %f) -> (%f, %f, %f), %f\n", signx1, signy1, 0.0, (u1 + pm[1]*signy1), (pm[4]*signx1+pm[5]*signy1+pm[7]), (u2 + pm[9]*signy1), utest);

    // printf("\tidx: %d, O:(%d(%f), %d(%f)) = %f, Ur: [%d~%d]\n", idx, ic, ox, il, oy, vol[idx], int(minu), int(maxu));

    for (int ti = 0; ti <= (maxu - minu); ++ti, sx += puv.x, sy += puv.y) {
        // entry/exit points of the (pixel-thick) beamlet along the u-bin
        float ax = sx - puv.x * 0.5f;
        float ay = sy - puv.y * 0.5f;
        float bx = sx + puv.x * 0.5f;
        float by = sy + puv.y * 0.5f;

        // intersection parameters back onto the line through (px,py)->(sx,sy)
        float num  = ( (ox-px)*(sx-px) + (oy-py)*(sy-py) );
        float dena = ( (ax-px)*(sx-px) + (ay-py)*(sy-py) );
        float denb = ( (bx-px)*(sx-px) + (by-py)*(sy-py) );
        float ra = num / dena;
        float rb = num / denb;

        float ax1 = px + (ax-px) * ra;
        float ay1 = py + (ay-py) * ra;
        float bx1 = px + (bx-px) * rb;
        float by1 = py + (by-py) * rb;

        // angular term for hex “basis” widths
        float theta = atan2f(sx - px, sy - py);
        double va = (double)dt * sin((double)theta);
        double vb = (double)dt * sin((double)theta + 2.0 * PI / 3.0);
        double vc = (double)dt * sin((double)theta + 4.0 * PI / 3.0);

        // path-length blur along the ray segment
        double vblur = -sqrt( (double)(bx1-ax1)*(bx1-ax1) + (double)(by1-ay1)*(by1-ay1) );

        // signed distance y from (ax1) to voxel center
        double y = sqrt( (double)(ax1-ox)*(ax1-ox) + (double)(ay1-oy)*(ay1-oy) );
        if ( (ay - py)*(ox - px) < (oy - py)*(ax - px) )
            y = -y;

        // if ( fabs(y) > (fabs(va) + fabs(vb) + fabs(vc) + fabs(vblur)) )
        //     assert(0 && "y out of bounds"); // should never happen

        double conv = linear_convolution_1d(y, vblur, va, vb, vc);

        // printf("[Conv %d %d %d] #Cell: %d, theta: %.12f, va: %.12f, vb: %.12f, vc: %.12f, vblur: %.12f, y: %.12f, conv: %lf\n", ip, ic, il, minu + ti, (double)theta, (double)va, (double)vb, (double)vc, (double)vblur, (double)y, conv);
        if (conv > 0.0) {
            // area factor uses hex dx,dy derived from dt
            double val = conv * C * (double)dx * (double)dy;
            int idxu = (minu + ti) * np + ip;
            if (idxu >= 0 && idxu < (nu * np))
                atomicAdd(proj + idxu, val);
        }
    }
}


__global__ void hl_backward_projection(const double *proj, double *vol, const float *vecs, const float *pm, int3 n3xyz, float3 d3xyz, int nu, float3 src, float3 puv, float3 dtv, int z_size) {
    int ic = (blockIdx.x * blockDim.x) + threadIdx.x;
    // int blocky = (blockIdx.y * blockDim.y) + threadIdx.y;
    int il = (blockIdx.y * blockDim.y) + threadIdx.y;

    // int y_start = blocky * z_size;
    // int y_end = y_start + z_size;
    // y_end = min(y_end, n3xyz.y);
    // if(y_end <= y_start) return;

    int nc = n3xyz.x, nl = n3xyz.y;

    if (ic >= nc) return;
    if (il >= nl) return;
    if ((il & 1) && ic == (nc - 1)) return;

    float dt = d3xyz.y * 2.0f;
    float dx = dt * (float)SQRT3;
    float dy = dt * 0.5f;

    float nl2 = 0.5f * (nl - 1);
    float nc2 = 0.5f * (nc - 1);
    float nc3 = 0.5f * (nc - 2);

    float oy = (il - nl2) * dy;
    float ox = ((il & 1) ? (ic - nc3) : (ic - nc2)) * dx;

    float px = src.x, py = src.y;

    // printf("P:(%f, %f), O:(%d(%f), %d(%f))\n", px, py, ic, ox, il, oy);

    int idx = il * nc + ic;
    // double C = vol[idx];

    float eps = 1e-7f;
    bool singular = fabsf(puv.x - puv.y) < eps;

    // float ic_eff = (float)ic + ((il & 1) ? 0.5f : 0.0f);

    int minu, maxu;
    hex_u_bounds_6_safe(pm, ic, il, nu, minu, maxu);

    float sx = dtv.x + puv.x * (minu - 0.5f * nu + 0.5f);
    float sy = dtv.y + puv.y * (minu - 0.5f * nu + 0.5f);

    (void)vecs;

   

    // printf("BL: (%f, %f, %f) -> (%f, %f, %f), %f\n", signx1, signy1, 0.0, (u1 + pm[1]*signy1), (pm[4]*signx1+pm[5]*signy1+pm[7]), (u2 + pm[9]*signy1), utest);

    // printf("\tidx: %d, O:(%d(%f), %d(%f)) = %f, Us: [%f, %f, %f, %f], Ur: [%d~%d]\n", idx, ix, ox, iy, oy, vol[idx], us[0], us[1], us[2], us[3], int(minu), int(maxu));

    double acc = 0.0;

    for (int ti = 0; ti <= (maxu - minu); ++ti, sx += puv.x, sy += puv.y) {
        float ax = sx - puv.x * 0.5f;
        float ay = sy - puv.y * 0.5f;
        float bx = sx + puv.x * 0.5f;
        float by = sy + puv.y * 0.5f;

        float num  = ( (ox-px)*(sx-px) + (oy-py)*(sy-py) );
        float dena = ( (ax-px)*(sx-px) + (ay-py)*(sy-py) );
        float denb = ( (bx-px)*(sx-px) + (by-py)*(sy-py) );
        float ra = num / dena;
        float rb = num / denb;

        float ax1 = px + (ax-px) * ra;
        float ay1 = py + (ay-py) * ra;
        float bx1 = px + (bx-px) * rb;
        float by1 = py + (by-py) * rb;

        float da = hypotf(ax1 - ox, ay1 - oy);
        float db = hypotf(bx1 - ox, by1 - oy);
        if (da > dt && db > dt && (ax1-ox)*(bx1-ox) >= 0.0f && (ay1-oy)*(by1-oy) >= 0.0f)
            continue;

        float theta = atan2f(sx - px, sy - py);
        double va = (double)dt * sin((double)theta);
        double vb = (double)dt * sin((double)theta + 2.0 * PI / 3.0);
        double vc = (double)dt * sin((double)theta + 4.0 * PI / 3.0);

        double vblur = -sqrt( (double)(bx1-ax1)*(bx1-ax1) + (double)(by1-ay1)*(by1-ay1) );

        double y = sqrt( (double)(ax1-ox)*(ax1-ox) + (double)(ay1-oy)*(ay1-oy) );
        if ( (ay - py)*(ox - px) < (oy - py)*(ax - px) )
            y = -y;

        double conv = linear_convolution_1d(y, vblur, va, vb, vc);
        if (conv == conv && conv > 0.0) { // finite & positive
            int idxu = (minu + ti);
            if (idxu >= 0 && idxu < nu) {
                acc += conv * proj[idxu] * (double)dx * (double)dy;
            }
        }
    }

    if (acc != 0.0)
        atomicAdd(vol + idx, acc);
}

__global__ void hl_matrix_generate(double *mat, int3 n3xyz, double3 d3xyz, int nu, double3 src, double3 puv, double3 dtv) { // nz = 1, dz = 0, src.z = 0
    //int ip = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iu = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iu >= nu) return;

    int nl, nc;
    int row;
    double nl2, nc2, nc3;
    double dt, dx, dy;
    int i, j;
    double px, py;
    double sx, sy, ax, ay, bx, by;
    double ox, oy;
    double theta;
    double va, vb, vc, vblur, y;

    nl = n3xyz.x, nc = n3xyz.y;
    nl2 = 0.5 * (nl-1), nc2 = 0.5 * (nc-1), nc3 = 0.5 * (nc-2);
    dt = d3xyz.x;
    dx = dt*SQRT3, dy = dt*0.5;
    px = src.x, py = src.y;
    
    sx = dtv.x + puv.x * (iu - 0.5 * nu + 0.5);
    sy = dtv.y + puv.y * (iu - 0.5 * nu + 0.5);
    ax = sx - puv.x * 0.5;
    ay = sy - puv.y * 0.5;
    bx = sx + puv.x * 0.5;
    by = sy + puv.y * 0.5;

    theta = atan2(sx-px, sy-py);

    va = dt*sin(theta);
    vb = dt*sin(theta + PI*2/3);
    vc = dt*sin(theta + PI*4/3);
    
    // TODO: optimize the range of j
    for(i = 0; i < nl; ++i) {
        for(j = 0; j < nc; ++j) {
            oy = (i - nl2) * dy;
            if(i&1) {
                if(j == nc-1) continue;
                ox = (j - nc3) * dx;
            }
            else {
                ox = (j - nc2) * dx;
            }

            row = i*nc + j;

            double ra = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((ax-px)*(sx-px) + (ay-py)*(sy-py));
            double rb = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((bx-px)*(sx-px) + (by-py)*(sy-py));
            double ax1 = px + (ax-px) * ra, ay1 = py + (ay-py) * ra;
            double bx1 = px + (bx-px) * rb, by1 = py + (by-py) * rb;
            //y = fabs((sy-py)*ox - (sx-px)*oy + sx*py - sy*px) / sqrtf64((sx-px)*(sx-px) + (sy-py)*(sy-py));

#ifdef DEBUG
            printf("[Slot %d %d %d] ra: %lf, rb: %lf, s:(%lf %lf), p:(%lf, %lf), o:(%lf %lf), a:(%lf %lf), b:(%lf %lf), a1:(%lf %lf), b1:(%lf %lf)\n", iu, i, j, ra, rb, sx, sy, px, py, ox, oy, ax, ay, bx, by, ax1, ay1, bx1, by1);
#endif

            if (sqrt((ax1-ox)*(ax1-ox) + (ay1-oy)*(ay1-oy)) > dt && sqrt((bx1-ox)*(bx1-ox) + (by1-oy)*(by1-oy)) > dt && (ax1-ox)*(bx1-ox) >= 0 && (ay1-oy)*(by1-oy) >= 0)
                continue;

            vblur = -sqrt((bx1-ax1) * (bx1-ax1) + (by1-ay1) * (by1-ay1));
            y = sqrt((ax1-ox) * (ax1-ox) + (ay1-oy) * (ay1-oy));

            if (y > fabs(va) + fabs(vb) + fabs(vc) + fabs(vblur))
                continue;

            if((ay-py) * (ox-px) < (oy-py) * (ax-px)) 
                y = -y;

            double conv = linear_convolution_1d(y, vblur, va, vb, vc);

#ifdef DEBUG
            printf("[Conv %d %d %d] theta: %.12lf, va: %.12lf, vb: %.12lf, vc: %.12lf, vblur: %.12lf, y: %.12lf, conv: %lf\n", iu, i, j, theta, va, vb, vc, vblur, y, conv);
#endif

            mat[iu*nl*nc + row] = conv;
        }
    }
}

A_Hl::A_Hl(GeoData *geo) {
    geodata = geo;

    ngrid = 1;

    nblock = (geo->nuv.x + (ngrid-1)) / ngrid;


#ifndef DEBUG
    vblock = dim3(16, 8);
#else
    vblock = dim3(1, 1);
#endif

    int bx = (geo->nxyz.x + vblock.x - 1) / vblock.x;
    int by = (geo->nxyz.y + (vblock.y - 1)) / (vblock.y);
    int byz = (geo->nxyz.y + (vblock.y * Z_SIZE - 1)) / (vblock.y * Z_SIZE);

    vgrid = dim3(bx, by);
    vgrid_z = dim3(bx, byz);

    // vecs = new Matrix(geo->np * geo->nuv.x, 5);
    // vecs = new Matrix(geo->np * geo->nuv.x, 2);
    // vecs->allocateMemory();

    // generate_vectors((*vecs)[0], geo);
    // vecs->copyHostToDevice();

    printf("%d %d %d %d\n", vgrid.x, vgrid.y, vblock.x, vblock.y);
}

A_Hl::~A_Hl() {

}

void A_Hl::project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
#endif
        hl_forward_projection<<<vgrid, vblock>>>(proj(0), vol(0), nullptr, geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->nuv.x,
                                                        make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]), Z_SIZE, geodata->np, p);
    
        cudaDeviceSynchronize();
        }
}

void A_Hl::back_project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
#endif

        hl_backward_projection<<<vgrid, vblock>>>(proj(int(p * geodata->nuv.x)), vol(0), nullptr, geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->nuv.x,
                                                        make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]), 1);
    }
}

extern "C" {
/*
    int hex_matrix_generation(int np, int nu, int nl, int nc, double du, double dt, double lsd, double lso, double *spmat, int buffer_size) {
        GeoData *geo = new GeoData(nl, nc, 1, nu, 1, np, dt, dt, 0, du, 0);
        geo->geo_init_example(lsd, lso, 0.0f, PI*2 * (np-1)/np);

        MatrixD mat(nu, nl*nc);
        mat.allocateMemory();

        int idx = 0;

        int ngrid = 16;
        int nblock = (geo->nuv.x + (ngrid-1)) / ngrid;

        for(int p=0; p<np; p++){

            std::fill(mat[0], mat[nu*nl*nc], 0.0);
            mat.copyHostToDevice();

            hl_matrix_generate<<<ngrid, nblock>>>(mat(0), geo->nxyz, geo->dxyz, geo->nuv.x,
                                                    make_double3(*geo->srcs[p*3], *geo->srcs[p*3+1], *geo->srcs[p*3+2]),
                                                    make_double3(*geo->puvs[p*3], *geo->puvs[p*3+1], *geo->puvs[p*3+2]),
                                                    make_double3(*geo->dtvs[p*3], *geo->dtvs[p*3+1], *geo->dtvs[p*3+2]));
            mat.copyDeviceToHost();

            for(int u=0; u<nu; ++u) {
                for(int i=0; i<nl; ++i) {
                    for(int j=0; j<nc; ++j) {
                        if(*mat[u*nl*nc + i*nc + j] != 0){
                            if(idx == buffer_size)
                                assert(0);

                            spmat[3*idx] = p*nu + u;
                            spmat[3*idx+1] = i*nc + j;
                            spmat[3*idx+2] = *mat[u*nl*nc + i*nc + j];
                            ++idx;
                        }
                    }
                }
            }
        }

        delete geo;

        return idx;
    }
*/

    A_Hl *hl_init(int nl, int nc, int np, int nu, double dt, double du, double lsd, double lso, double *angles, double *dz, double *drho){
        GeoData *geo = new GeoData(nc, nl, 1, nu, 1, np, dt * SQRT3, dt * 0.5f, 1, du, 1);
        geo->geo_init_angles(lsd, lso, angles, dz, drho);
        geo->initialize_projection_matrix();

        A_Hl *hl_layer = new A_Hl(geo);

        return hl_layer;
    }

    int hl_forward_projection(double *b, double *x, A_Hl *hl_layer) {
        int nc = hl_layer->geodata->nxyz.x;
        int nl = hl_layer->geodata->nxyz.y;

        int np = hl_layer->geodata->np;
        int nu = hl_layer->geodata->nuv.x;

        int cnt = 0;

        MatrixD vol(nc, nl);
        MatrixD proj(np, nu);

        vol.allocateMemory();
        proj.allocateMemory();

        memset(proj[0], 0, np*nu*sizeof(double));
        for(int ic=0; ic<nc; ++ic)
            for(int il=0; il<nl; ++il){
                *(vol[il * nc + ic]) = x[ic*nl+il];
                // printf("%d %d %f %f\n",il * nc + ic, ic*nl+il, x[ic*nl+il], *(vol[il * nc + ic]));
            }

        vol.copyHostToDevice();
        proj.copyHostToDevice();

        cudaMemcpyToSymbol(conv_count, &cnt, sizeof(int));

        hl_layer->project(vol, proj, 1.0);

        cudaMemcpyFromSymbol(&cnt, conv_count, sizeof(int));
        // fprintf(stderr ,"Conv count: %d\n", cnt);

        cudaDeviceSynchronize();

        proj.copyDeviceToHost();

        for(int ip=0; ip<np; ++ip)
            for(int iu=0; iu<nu; ++iu)
                b[ip * nu + iu] = *(proj[iu * np + ip]);

        return cnt;
    }

    void hl_backward_projection(double *b, double *x, A_Hl *hl_layer) {
        int nc = hl_layer->geodata->nxyz.x;
        int nl = hl_layer->geodata->nxyz.y;

        int np = hl_layer->geodata->np;
        int nu = hl_layer->geodata->nuv.x;

        // int cnt = 0;

        MatrixD vol(nc, nl);
        MatrixD proj(np, nu);

        vol.allocateMemory();
        proj.allocateMemory();

        memset(vol[0], 0, nc*nl*sizeof(double));
        memcpy(proj[0], b, np*nu*sizeof(double));

        vol.copyHostToDevice();
        proj.copyHostToDevice();

        hl_layer->back_project(vol, proj, 1.0);

        // fprintf(stderr, "b, x, %p, %p, %lf", b, x, b[10000]);

        cudaDeviceSynchronize();

        vol.copyDeviceToHost();

        // printf("%p, %p, %p, %lf\n", vol[0], vol(0), x, *(vol[100]));

        for(int ic=0; ic<nc; ++ic)
            for(int il=0; il<nl; ++il)
                x[ic*nl+il] = *(vol[il * nc + ic]);
    }
}