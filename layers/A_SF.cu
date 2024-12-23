#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "A_SF.hh"
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

__device__ static void gamma_calculate(float s1, float s2, float *us, float *gamma) {
    float tmp = 0.0f;
    float b1, b2;

    b1 = fmaxf(s1, us[0]);
    b2 = fminf(s2, us[1]);

    if(b2 > b1){
        float _a = b1 - us[0];
        float _b = b2 - b1;
        tmp =  (_b * _b + 2 * _a * _b) / (2 * (us[1] - us[0]));
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
        float _b = b2 - b1;
        float _c = us[3] - b2;
        tmp = (_b * _b + 2 * _b * _c) / (2 * (us[3] - us[2]));
    }
    else{
        tmp = 0.0f;
    }
    if(tmp == tmp) gamma[0] += tmp ;
    
    gamma[0] *= (2.0f / ( ((us[3]-us[0])+(us[2]-us[1])) ));
}

// Fan Beam only
__global__ static void SF_forward_projection(double *proj, const double *vol, const float *pm, int3 n3xyz, float3 d3xyz, int nu, float3 src, float3 puv, float3 dtv, float lsd, int z_size, int np, int ip) {
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
    // float u1, u2, u3, u4;
    float us[4] = {0.0};
    float signx1, signx2, signy1, signy2;

    float eps = 1e-7f;
    float C;

    // double a0s, sb0, lsp;
    // double vblur, y, r1;
    double val;

    signx1 = ix - 0.5f;
    signx2 = ix + 0.5f;

    bool singular = fabs(puv.x - puv.y) < eps;

    // printf("\n");

    // u1 = pm[0]*signx1 + pm[3];
    // u2 = pm[8]*signx1 + pm[11];

    // u3 = pm[0]*signx2 + pm[3];
    // u4 = pm[8]*signx2 + pm[11];

    // for(int iy = y_start; iy < y_end; ++ iy, oy += dy) {
        idx = iy*nx+ix;
        C = vol[idx];

        signy1 = iy - 0.5f;
        signy2 = iy + 0.5f;

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

        // printf("\tidx: %d, O:(%d(%f), %d(%f)) = %f, Us: [%f, %f, %f, %f], Ur: [%d~%d]\n", idx, ix, ox, iy, oy, vol[idx], us[0], us[1], us[2], us[3], int(minu), int(maxu));

        val = 0;

        float weight = rsqrtf( (d3xyz.x*(ix - nx/2)-src.x)*(d3xyz.x*(ix - nx/2)-src.x) + (d3xyz.y*(iy - ny/2)-src.y)*(d3xyz.y*(iy - ny/2)-src.y)) * lsd;

        for (int ti = 0; ti < maxu - minu + 1; ++ ti) {
            idxu = (minu + ti) * np + ip;

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
            float gamma = 0.0f;

            gamma_calculate(minu + ti, minu + ti + 1.0f, us, &gamma);

            val = weight * gamma * C * dx * dy;
            // if (idx == 4)
            //     printf("idx: %d, O:(%d, %d) = %f, Us: [%f, %f, %f, %f], U_Index: %d, Gamma: %f\n", idx, ix, iy, vol[idx], us[0], us[1], us[2], us[3], minu + ti, gamma);
            
            if(idxu < np * nu && idxu >= 0 && val == val && gamma > eps)
                atomicAdd(proj+idxu, val);
                // proj[idxu] += val;
        }
    // }
}


__global__ static void SF_backward_projection(const double *proj, double *vol, const float *pm, int3 n3xyz, float3 d3xyz, int nu, float3 src, float3 puv, float3 dtv, float lsd, int z_size) {
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
    // float u1, u2, u3, u4;
    float us[4] = {0.0};
    float signx1, signx2, signy1, signy2;

    float eps = 1e-7f;

    // double a0s, sb0, lsp;
    // double vblur, y, r1;
    double val;
    // oy = (iy-0.5f * (ny-1)) * dy;

    signx1 = ix - 0.5f;
    signx2 = ix + 0.5f;

    bool singular = fabs(puv.x - puv.y) < eps;

    // u1 = pm[0]*signx1 + pm[3];
    // u2 = pm[8]*signx1 + pm[11];

    // u3 = pm[0]*signx2 + pm[3];
    // u4 = pm[8]*signx2 + pm[11];

    // printf("P:(%f, %f)\n", px, py);

    for(int iy = y_start; iy < y_end; ++ iy) {
        idx = iy*nx+ix;

        signy1 = iy - 0.5f;
        signy2 = iy + 0.5f;

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

        float weight = rsqrtf( (d3xyz.x*(ix - nx/2)-src.x)*(d3xyz.x*(ix - nx/2)-src.x) + (d3xyz.y*(iy - ny/2)-src.y)*(d3xyz.y*(iy - ny/2)-src.y)) * lsd;

        // printf("\tidx: %d, O:(%d(%f), %d(%f)) = %f, Us: [%f, %f, %f, %f], Ur: [%d~%d]\n", idx, ix, ox, iy, oy, vol[idx], us[0], us[1], us[2], us[3], int(minu), int(maxu));

        val = 0;

        for (int ti = 0; ti < maxu - minu + 1; ++ ti) {
            idxu = minu + ti;

            float gamma = 0.0f;

            gamma_calculate(idxu, idxu + 1.0f, us, &gamma);

            if(idxu < nu && idxu >= 0 && gamma == gamma && gamma > eps)
                val += weight * gamma * proj[idxu] * dx * dy;
        }

        atomicAdd(vol+idx, val);
    }
}

A_SF::A_SF(GeoData *geo)
{
    geodata = geo;

    ngrid = 1;

    nblock = (geo->nuv.x + (ngrid-1)) / ngrid;

    vblock = dim3(16, 8);
    // vblock = dim3(1, 1);

    int bx = (geo->nxyz.x + vblock.x - 1) / vblock.x;
    int by = (geo->nxyz.y + (vblock.y - 1)) / (vblock.y);
    int byz = (geo->nxyz.y + (vblock.y * Z_SIZE - 1)) / (vblock.y * Z_SIZE);

    vgrid = dim3(bx, by);
    vgrid_z = dim3(bx, byz);

    printf("%d %d %d %d\n", vgrid.x, vgrid.y, vblock.x, vblock.y);
}

A_SF::~A_SF()
{
}

void A_SF::project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
#endif
        SF_forward_projection<<<vgrid_z, vblock>>>(proj(0), vol(0), geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->nuv.x,
                                                        make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]), *geodata->lsds[p], Z_SIZE, geodata->np, p);
    }
}

void A_SF::back_project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
#endif

        SF_backward_projection<<<vgrid, vblock>>>(proj(int(p * geodata->nuv.x)), vol(0), geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3(geodata->dxyz.x, geodata->dxyz.y, geodata->dxyz.z), geodata->nuv.x,
                                                        make_float3(*geodata->srcs[p*3], *geodata->srcs[p*3+1], *geodata->srcs[p*3+2]),
                                                        make_float3(*geodata->puvs[p*3], *geodata->puvs[p*3+1], *geodata->puvs[p*3+2]),
                                                        make_float3(*geodata->dtvs[p*3], *geodata->dtvs[p*3+1], *geodata->dtvs[p*3+2]), *geodata->lsds[p], 1);
    }
}


extern "C" {
    A_SF *SF_init(int nx, int ny, int np, int nu, double dx, double dy, double du, double lsd, double lso, double *angles){
        GeoData *geo = new GeoData(nx, ny, 1, nu, 1, np, dx, dy, 1, du, 1);
        geo->geo_init_angles(lsd, lso, angles);
        geo->initialize_projection_matrix();

        A_SF *SF_layer = new A_SF(geo);

        return SF_layer;
    }

    int SF_forward_projection(double *b, double *x, A_SF *SF_layer) {
        int nx = SF_layer->geodata->nxyz.x;
        int ny = SF_layer->geodata->nxyz.x;

        int np = SF_layer->geodata->np;
        int nu = SF_layer->geodata->nuv.x;

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

        SF_layer->project(vol, proj, 1.0);

        cudaMemcpyFromSymbol(&cnt, conv_count, sizeof(int));
        // fprintf(stderr ,"Conv count: %d\n", cnt);

        cudaDeviceSynchronize();

        proj.copyDeviceToHost();

        for(int ip=0; ip<np; ++ip)
            for(int iu=0; iu<nu; ++iu)
                b[ip * nu + iu] = *(proj[iu * np + ip]);

        return cnt;
    }

    void SF_backward_projection(double *b, double *x, A_SF *SF_layer) {
        int nx = SF_layer->geodata->nxyz.x;
        int ny = SF_layer->geodata->nxyz.x;

        int np = SF_layer->geodata->np;
        int nu = SF_layer->geodata->nuv.x;

        MatrixD vol(nx, ny);
        MatrixD proj(np, nu);

        vol.allocateMemory();
        proj.allocateMemory();

        memset(vol[0], 0, nx*ny*sizeof(double));
        memcpy(proj[0], b, np*nu*sizeof(double));

        vol.copyHostToDevice();
        proj.copyHostToDevice();

        SF_layer->back_project(vol, proj, 1.0);

        // fprintf(stderr, "b, x, %p, %p, %lf", b, x, b[10000]);

        cudaDeviceSynchronize();

        vol.copyDeviceToHost();

        // printf("%p, %p, %p, %lf\n", vol[0], vol(0), x, *(vol[100]));

        for(int ix=0; ix<nx; ++ix)
            for(int iy=0; iy<ny; ++iy)
                x[ix*ny+iy] = *(vol[iy * nx + ix]);
    }
}