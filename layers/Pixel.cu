#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "Bilinear.hh"
#include "../utils/geo.hh"
#include "../utils/matrix_double.hh"
#include "../utils/exception.hh"

#include "spline_conv.hh"

__device__ static double linear_convolution_1d(double y, double h1, double h2, double h3) 
{
    int tot = 0;
    double hs[3];
    if(fabs(h1) > 1e-8) hs[tot++] = h1;
    if(fabs(h2) > 1e-8) hs[tot++] = h2;
    if(fabs(h3) > 1e-8) hs[tot++] = h3;

    //printf("%d: %lf, %lf, %lf, %lf, %lf\n", tot, hs[0], hs[1], hs[2], hs[3], hs[4]);

    switch(tot) {
        case 3:
            return forward_difference_3x_1(y, hs[0], hs[1], hs[2]) / 2.0;
            break;
        case 2:
            return forward_difference_2x_1(y, hs[0], hs[1]);
            break;
        default:
            assert(0);
            break;
    }
}

__global__ void pixel_matrix_generate(double *mat, int3 n3xyz, double3 d3xyz, int nu, double3 src, double3 puv, double3 dtv) { // nz = 1, dz = 0, src.z = 0
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
    double zx, zy;
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
    vx = fabs(dx * cost), vy = fabs(dy * sint);

    if(py < sy) vx = -vx;
    if(px < sx) vy = -vy;
    
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

            //printf("[Slot %d %d %d] ra: %lf, rb: %lf, s:(%lf %lf), p:(%lf, %lf), o:(%lf %lf), a:(%lf %lf), b:(%lf %lf), a1:(%lf %lf), b1:(%lf %lf)\n", iu, i, j, ra, rb, sx, sy, px, py, ox, oy, ax, ay, bx, by, ax1, ay1, bx1, by1);

            if (fabs(ax1-ox) > dx && fabs(bx1-ox) > dx && fabs(ay1-oy) > dy && fabs(by1-oy) > dy)
                continue;

            double OM_dot_SP = -0.5*dx*(px-sx) + 0.5*dy*(py-sy);
            double NM_x = OM_dot_SP*(px-sx)/((sx-px)*(sx-px)+(sy-py)*(sy-py));
            double NM_y = OM_dot_SP*(py-sy)/((sx-px)*(sx-px)+(sy-py)*(sy-py));

            zx = ox - 0.5 * dx - NM_x;
            zy = oy + 0.5 * dy - NM_y;

            vblur = -sqrt((bx1-ax1) * (bx1-ax1) + (by1-ay1) * (by1-ay1));
            //y = sqrt((ax1-ox) * (ax1-ox) + (ay1-oy) * (ay1-oy));
            y = sqrt((ax1-zx) * (ax1-zx) + (ay1-zy) * (ay1-zy));

            // if((ay-py) * (ox-px) > (oy-py) * (ax-px)) 
            //     y = -y;

            if (y > fabs(vx) + fabs(vy) + fabs(vblur))
                continue;

            if((ay-py) * (zx-px) < (zy-py) * (ax-px)) 
                y = -y;

            double conv = linear_convolution_1d(y, vx, vy, vblur);

            //printf("[Conv %d %d %d] NM: (%lf, %lf), z: (%lf, %lf), theta: %lf, vx: %lf, vy: %lf, vblur: %lf, y: %lf, conv: %lf\n", iu, i, j, NM_x, NM_y, zx, zy, theta, vx, vy, vblur, y, conv);

            if(conv > 0)
                mat[iu*nx*ny + row] = conv;
        }
    }
}


extern "C" {
    int pixel_matrix_generation(int np, int nu, int nx, int ny, double dx, double dy, double du, double lsd, double lso, double *spmat, int buffer_size) {
        GeoData *geo = new GeoData(nx, ny, 1, nu, 1, np, dx, dy, 0, du, 0);
        geo->geo_init_example(lsd, lso, 0.0f, PI*2 * (np-1)/np);

        MatrixD mat(nu, nx*ny);
        mat.allocateMemory();

        int idx = 0;

        int ngrid = 16;
        int nblock = (geo->nuv.x + (ngrid-1)) / ngrid;

        for(int p=0; p<np; p++){

            std::fill(mat[0], mat[nu*nx*ny], 0.0);
            mat.copyHostToDevice();

            pixel_matrix_generate<<<ngrid, nblock>>>(mat(0), geo->nxyz, geo->dxyz, geo->nuv.x,
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
}