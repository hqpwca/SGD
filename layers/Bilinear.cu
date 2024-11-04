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

__device__ double base_function_5x(double y)
{
    if(y > 0) return y*y*y*y;
    else return 0;
}

__device__ double base_function_3x(double y)
{
    if(y > 0) return y*y;
    else return 0;
}

__device__ double forward_difference_3x_3(double y, double h) {
    double res;
    res = base_function_3x(y) - base_function_3x(y-h);
    return res/h;
}

__device__ double forward_difference_3x_2(double y, double h, double h2) {
    double res;
    res = forward_difference_3x_3(y, h) - forward_difference_3x_3(y-h2, h);
    return res/h2;
}

__device__ double forward_difference_3x_1(double y, double h, double h2, double h3) {
    double res;
    res = forward_difference_3x_2(y, h, h2) - forward_difference_3x_2(y-h3, h, h2);
    return res/h3;
}

__device__ double forward_difference_5x_5(double y, double h) {
    double res;
    res = base_function_5x(y) - base_function_5x(y-h);
    return res/h;
}

__device__ double forward_difference_5x_4(double y, double h, double h2) {
    double res;
    res = forward_difference_5x_5(y, h) - forward_difference_5x_5(y-h2, h);
    return res/h2;
}

__device__ double forward_difference_5x_3(double y, double h, double h2, double h3) {
    double res;
    res = forward_difference_5x_4(y, h, h2) - forward_difference_5x_4(y-h3, h, h2);
    return res/h3;
}

__device__ double forward_difference_5x_2(double y, double h, double h2, double h3, double h4) {
    double res;
    res = forward_difference_5x_3(y, h, h2, h3) - forward_difference_5x_3(y-h4, h, h2, h3);
    return res/h4;
}

__device__ double forward_difference_5x_1(double y, double h, double h2, double h3, double h4, double h5) {
    double res;
    res = forward_difference_5x_2(y, h, h2, h3, h4) - forward_difference_5x_2(y-h5, h, h2, h3, h4);
    return res/h5;
}

__device__ double linear_convolution_1d(double y, double h1, double h2, double h3, double h4, double h5) 
{
    int tot = 0;
    double hs[5];
    if(fabs(h1) > 1e-8) hs[tot++] = h1;
    if(fabs(h2) > 1e-8) hs[tot++] = h2;
    if(fabs(h3) > 1e-8) hs[tot++] = h3;
    if(fabs(h4) > 1e-8) hs[tot++] = h4;
    if(fabs(h5) > 1e-8) hs[tot++] = h5;

    //printf("%d: %lf, %lf, %lf, %lf, %lf\n", tot, hs[0], hs[1], hs[2], hs[3], hs[4]);

    switch(tot) {
        case 3:
            return forward_difference_3x_1(y, hs[0], hs[1], hs[2]) / 2.0;
            break;
        case 5:
            return forward_difference_5x_1(y, hs[0], hs[1], hs[2], hs[3], hs[4]) / 24.0;
            break;
        default:
            assert(0);
            break;
    }
}

// Fan Beam only
__global__ void bilinear_forward_projection(double *proj, const float *vol, int3 n3xyz, double3 d3xyz, int nu, double3 src, double3 puv, double3 dtv) { // nz = 1, dz = 0, src.z = 0
    //int ip = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iu = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iu >= nu) return;

    int nx, ny;
    double nx2, ny2;
    double dx, dy;
    int i, j;
    double px, py;
    double sx, sy, ax, ay, bx, by;
    double ox, oy;
    double theta, sint, cost;
    double vx, vy, vblur, y;

    double V = 0.0f;

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
    
    double sum = 0.0;
    // TODO: optimize the range of j
    for(i = 0; i < nx; ++i) {
        for(j = 0; j < ny; ++j) {
            ox = (i-nx2) * dx;
            oy = (j-ny2) * dy;

            double ra = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((ax-px)*(sx-px) + (ay-py)*(sy-py));
            double rb = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / ((bx-px)*(sx-px) + (by-py)*(sy-py));
            double ax1 = px + (ax-px) * ra, ay1 = py + (ay-py) * ra;
            double bx1 = px + (bx-px) * rb, by1 = py + (by-py) * rb;
            //y = fabs((sy-py)*ox - (sx-px)*oy + sx*py - sy*px) / sqrtf64((sx-px)*(sx-px) + (sy-py)*(sy-py));

            //printf("[Slot %d %d %d] ra: %lf, rb: %lf, s:(%lf %lf), p:(%lf, %lf), o:(%lf %lf), a:(%lf %lf), b:(%lf %lf), a1:(%lf %lf), b1:(%lf %lf)\n", iu, i, j, ra, rb, sx, sy, px, py, ox, oy, ax, ay, bx, by, ax1, ay1, bx1, by1);

            if (fabs(ax1-ox) > dx && fabs(bx1-ox) > dx && fabs(ay1-oy) > dy && fabs(by1-oy) > dy)
                continue;

            vblur = sqrt((bx1-ax1) * (bx1-ax1) + (by1-ay1) * (by1-ay1));
            y = sqrt((ax1-ox) * (ax1-ox) + (ay1-oy) * (ay1-oy));

            if((ay-py) * (ox-px) > (oy-py) * (ax-px)) 
                y = -y;

            double conv = linear_convolution_1d(y, vx, -vx, vy, -vy, vblur);

            //printf("[Conv %d %d %d] theta: %lf, vx: %lf, vy: %lf, vblur: %lf, y: %lf, conv: %lf\n", iu, i, j, theta, vx, vy, vblur, y, conv);

            V = vol[j*nx + i] * conv;

            sum += V;
        }
    }
    proj[iu] = sum;
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

            //printf("[Slot %d %d %d] ra: %lf, rb: %lf, s:(%lf %lf), p:(%lf, %lf), o:(%lf %lf), a:(%lf %lf), b:(%lf %lf), a1:(%lf %lf), b1:(%lf %lf)\n", iu, i, j, ra, rb, sx, sy, px, py, ox, oy, ax, ay, bx, by, ax1, ay1, bx1, by1);

            if (fabs(ax1-ox) > dx && fabs(bx1-ox) > dx && fabs(ay1-oy) > dy && fabs(by1-oy) > dy)
                continue;

            vblur = sqrt((bx1-ax1) * (bx1-ax1) + (by1-ay1) * (by1-ay1));
            y = sqrt((ax1-ox) * (ax1-ox) + (ay1-oy) * (ay1-oy));

            if((ay-py) * (ox-px) > (oy-py) * (ax-px)) 
                y = -y;

            double conv = linear_convolution_1d(y, vx, -vx, vy, -vy, vblur);

            //printf("[Conv %d %d %d] theta: %lf, vx: %lf, vy: %lf, vblur: %lf, y: %lf, conv: %lf\n", iu, i, j, theta, vx, vy, vblur, y, conv);

            mat[iu*nx*ny + row] = conv;
        }
    }
}

__global__ void bilinear_backward_projection() {

}

Bilinear::Bilinear(GeoData *geo) {
    geodata = geo;

    ngrid = 1;
    nblock = (geo->nuv.x + (ngrid-1)) / ngrid;
}

Bilinear::~Bilinear() {

}

void Bilinear::project(Matrix &vol, MatrixD &proj, double weight) {
    for(int p=0; p<geodata->np; p++){
        bilinear_forward_projection<<<ngrid, nblock>>>(proj(int(p * geodata->nuv.x)), vol(0), geodata->nxyz, geodata->dxyz, geodata->nuv.x,
                                                        make_double3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_double3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_double3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]));
        cudaDeviceSynchronize();
        //std::cout << p << std::endl;
    }
}

void Bilinear::matrix(MatrixD &mat) {
    for(int p=0; p<geodata->np; p++){
        bilinear_matrix_generate<<<ngrid, nblock>>>(mat(p*geodata->nuv.x*geodata->nxyz.x*geodata->nxyz.y), geodata->nxyz, geodata->dxyz, geodata->nuv.x,
                                                        make_double3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_double3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_double3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]));
        cudaDeviceSynchronize();
    }
}

void Bilinear::back_project(Matrix &vol, Matrix &proj, double weight) {

}

void Bilinear::project(Matrix &vol, Matrix &proj, double weight) {
}

Matrix& Bilinear::forward(cublasHandle_t &cublasH, Matrix &x) {
}

Matrix& Bilinear::back_prop(cublasHandle_t &cublasH, Matrix &od, float lr) {
}

extern "C" {
    void bilinear_matrix_generation(int np, int nu, int nx, int ny, double dx, double dy, double du, double lsd, double lso, double *nmat) {
        GeoData *geo = new GeoData(nx, ny, 1, nu, 1, np, dx, dy, 0, du, 0);
        geo->geo_init_example(lsd, lso, 0.0f, PI*2 * (np-1)/np);

        int row = geo->nxyz.y * geo->nxyz.x;
        int col = geo->np * geo->nuv.x;

        Bilinear *bl_layer = new Bilinear(geo);

        MatrixD mat(row, col);
        mat.allocateMemory();
        std::fill(mat[0], mat[row*col], 0.0);
        mat.copyHostToDevice();

        bl_layer->matrix(mat);

        mat.copyDeviceToHost();

        memcpy(nmat, mat[0], row*col*sizeof(double));

        delete bl_layer;
        delete geo;
    }
}