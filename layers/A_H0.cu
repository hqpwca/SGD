#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>

#include "A_H0.hh"
#include "../utils/geo.hh"
#include "../utils/matrix_double.hh"
#include "../utils/exception.hh"
#include "spline_conv.hh"

#define SQRT3 1.7320508075688772935274463415

// #define DEBUG

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

__global__ void h0_matrix_generate(double *mat, int3 n3xyz, double3 d3xyz, int nu, double3 src, double3 puv, double3 dtv) { // nz = 1, dz = 0, src.z = 0
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
    dx = dt*SQRT3, dy = dt*1.5;
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

            double conv = linear_convolution_1d(y, vblur, va, vb, 0) + linear_convolution_1d(y, vblur, vb, vc, 0) + linear_convolution_1d(y, vblur, va, vc, 0);

#ifdef DEBUG
            printf("[Conv %d %d %d] theta: %.12lf, va: %.12lf, vb: %.12lf, vc: %.12lf, vblur: %.12lf, y: %.12lf, conv: %lf\n", iu, i, j, theta, va, vb, vc, vblur, y, conv);
#endif

            mat[iu*nl*nc + row] = conv / 3;
        }
    }
}

A_H0::A_H0(GeoData *geo) {
    geodata = geo;

#ifdef DEBUG
    ngrid = 10;
#else
    ngrid = 16;
#endif

    nblock = (geo->nuv.x + (ngrid-1)) / ngrid;
}

A_H0::~A_H0()
{
}

void A_H0::project(Matrix &vol, MatrixD &proj, double weight)
{
}

void A_H0::back_project(Matrix &vol, MatrixD &proj, double weight)
{
}

extern "C" {
        int h0_matrix_generation(int np, int nu, int nl, int nc, double du, double dt, double lsd, double lso, double *spmat, int buffer_size) {
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

            h0_matrix_generate<<<ngrid, nblock>>>(mat(0), geo->nxyz, geo->dxyz, geo->nuv.x,
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
}

