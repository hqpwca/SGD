#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "A_B0.hh"
#include "../utils/geo.hh"
#include "../utils/matrix_double.hh"
#include "../utils/exception.hh"

#include "spline_conv.hh"

#define Z_SIZE 1

__device__ static int conv_count = 0;

__device__ static float linear_convolution_1d(float y, float v1, float v2, float v3)
{
    atomicAdd(&conv_count, 1);
    float vs[3];
    y = fabs(y - ( v1 + v2 + v3) / 2.0f);
    vs[0] = fabs(v1), vs[1] = fabs(v2), vs[2] = fabs(v3);
    sort3d(vs, vs+1, vs+2);
    if (vs[2] > 1e-7f) {
        //return forward_difference_3x_1(y, vs[0], vs[1], vs[2]) / 2.0;
        return univariate3dirboxspline_fast3<float>(y, vs);
    }
    else {
        //return forward_difference_2x_1(y, vs[0], vs[1]);
        return univariate3dirboxspline_fast2<float>(y, vs);
    }
}

__global__ void b0_matrix_generate(double *mat, int3 n3xyz, double3 d3xyz, int nu, double3 src, double3 puv, double3 dtv) { // nz = 1, dz = 0, src.z = 0
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

            // printf("[Slot %d %d %d] ra: %lf, rb: %lf, s:(%lf %lf), p:(%lf, %lf), o:(%lf %lf), a:(%lf %lf), b:(%lf %lf), a1:(%lf %lf), b1:(%lf %lf)\n", iu, i, j, ra, rb, sx, sy, px, py, ox, oy, ax, ay, bx, by, ax1, ay1, bx1, by1);

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

            float conv = linear_convolution_1d(y, vx, vy, vblur);

            // printf("[Conv %d %d %d] NM: (%lf, %lf), z: (%lf, %lf), theta: %lf, vx: %lf, vy: %lf, vblur: %lf, y: %lf, conv: %lf\n", iu, i, j, NM_x, NM_y, zx, zy, theta, vx, vy, vblur, y, conv);

            if(conv > 0)
                mat[iu*nx*ny + row] = conv;
        }
    }
}

__global__ void b0_forward_projection(double *proj, const double *vol, const double *vecs, const float *pm, int3 n3xyz, float3 d3xyz, int nu, float3 src, float3 puv, float3 dtv) {
    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    int blocky = (blockIdx.y * blockDim.y) + threadIdx.y;

    int y_start = blocky * Z_SIZE;
    int y_end = y_start + Z_SIZE;
    y_end = min(y_end, n3xyz.y);

    if(ix >= n3xyz.x || y_end <= y_start) return;

    int nx = n3xyz.x, ny = n3xyz.y;
    float dx = d3xyz.x, dy = d3xyz.y;
    float nx2 = 0.5f * (nx-1), ny2 = 0.5f * (ny-1);
    float maxu, minu;
    int idx, idxu;
    float ox, oy;
    float px, py;
    float sx, sy;
    float u1, u2, u3, u4;
    float us[4] = {0.0};
    float signx1, signx2, signy1, signy2;

    float eps = 1e-7;

    double vx, vy, a0s, sb0, lsp;
    double vblur, y, r1;
    double conv;
    double val;

    ox = (ix-nx2) * dx;
    oy = (y_start-ny2) * dy;

    px = src.x;
    py = src.y;

    signx1 = ix - 0.5f;
    signx2 = ix + 0.5f;

    bool singular = fabs(puv.x - puv.y) < eps;

    u1 = pm[0]*signx1 + pm[3];
    u2 = pm[8]*signx1 + pm[11];

    u3 = pm[0]*signx2 + pm[3];
    u4 = pm[8]*signx2 + pm[11];

    // printf("P:(%f, %f)\n", px, py);

    for(int iy = y_start; iy < y_end; ++ iy, oy += dy) {
        idx = iy * nx + ix;

        signy1 = iy - 0.5f;
        signy2 = iy + 0.5f;

        if (!singular) {
            us[0] = (u1 + pm[1]*signy1) / (u2 + pm[9]*signy1);
            us[1] = (u1 + pm[1]*signy2) / (u2 + pm[9]*signy2);
            us[2] = (u3 + pm[1]*signy1) / (u4 + pm[9]*signy1);
            us[3] = (u3 + pm[1]*signy2) / (u4 + pm[9]*signy2);
        }
        else {
            us[0] = ((u1 + pm[1]*signy1)/(u2 + pm[9]*signy1)*1.5f - 1.0f) / puv.x;
            us[1] = ((u1 + pm[1]*signy2)/(u2 + pm[9]*signy2)*1.5f - 1.0f) / puv.x;
            us[2] = ((u3 + pm[1]*signy1)/(u4 + pm[9]*signy1)*1.5f - 1.0f) / puv.x;
            us[3] = ((u3 + pm[1]*signy2)/(u4 + pm[9]*signy2)*1.5f - 1.0f) / puv.x;
        }

        // printf("BL: (%f, %f, %f) -> (%f, %f, %f), %f\n", signx1, signy1, 0.0, (u1 + pm[1]*signy1), (pm[4]*signx1+pm[5]*signy1+pm[7]), (u2 + pm[9]*signy1), utest);

        sort4<float>(us, us+1, us+2, us+3);

        minu = min(max(0, (int)floorf(us[0])), nu-1);
        maxu = min(max(0, (int)floorf(us[3])), nu-1);

        sx = dtv.x + puv.x * (minu - 0.5f * nu + 0.5f);
        sy = dtv.y + puv.y * (minu - 0.5f * nu + 0.5f);

        // printf("\tidx: %d, O:(%d(%f), %d(%f)) = %f, Us: [%f, %f, %f, %f], Ur: [%d~%d]\n", idx, ix, ox, iy, oy, vol[idx], us[0], us[1], us[2], us[3], int(minu), int(maxu));

        for (int ti = 0; ti < maxu - minu + 1; ++ ti, sx += puv.x, sy += puv.y) {
            idxu = minu + ti;

            vx  = vecs[       idxu];
            vy  = vecs[nu   + idxu];
            a0s = vecs[nu*2 + idxu];
            sb0 = vecs[nu*3 + idxu];
            lsp = vecs[nu*4 + idxu];

            r1 = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / (lsp*lsp);
            vblur = (a0s + sb0) * (-r1);
            y = crossg(sx-px, sy-py, ox-0.5f*dx-px, oy+0.5f*dy-py) / lsp;
            y += a0s * r1;

            // if (fabs(y) > fabs(vx) + fabs(vy) + fabs(vblur))
            //     continue;

            conv = linear_convolution_1d(-y, vx, vy, vblur);

            // printf("\t\t[U=%d] S:(%f, %f), P:(%f, %f), vx: %lf, vy: %lf, a0s: %lf, sb0: %lf, lsp: %lf, r1: %lf, vblur: %lf, y: %.12lf -> %.12lf [conv] %.12lf\n", idxu, sx, sy, px, py, vx, vy, a0s, sb0, lsp, r1, vblur, y, y+vblur, conv);

            val = conv * vol[idx] * dx * dy;
            
            if(idxu < nu && idxu >= 0 && val == val && conv > eps)
                atomicAdd(proj+idxu, val);
        }
    }
}

__global__ void b0_backward_projection(const double *proj, double *vol, const double *vecs, const float *pm, int3 n3xyz, float3 d3xyz, int nu, float3 src, float3 puv, float3 dtv) {
    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    int blocky = (blockIdx.y * blockDim.y) + threadIdx.y;

    int y_start = blocky * Z_SIZE;
    int y_end = y_start + Z_SIZE;
    y_end = min(y_end, n3xyz.y);

    if(ix >= n3xyz.x || y_end <= y_start) return;

    int &nx = n3xyz.x, &ny = n3xyz.y;
    float &dx = d3xyz.x, &dy = d3xyz.y;
    float nx2 = 0.5f * (nx-1), ny2 = 0.5f * (ny-1);
    float maxu, minu;
    int idx, idxu;
    float ox, oy;
    float &px = src.x, &py = src.y;
    float sx, sy;
    float u1, u2, u3, u4;
    float us[4] = {0.0};
    float signx1, signx2, signy1, signy2;

    float eps = 1e-7;

    double vx, vy, a0s, sb0, lsp;
    double vblur, y, r1;
    double conv;
    double val;

    ox = (ix-nx2) * dx;
    oy = (y_start-ny2) * dy;

    signx1 = ix - 0.5f;
    signx2 = ix + 0.5f;

    bool singular = fabs(puv.x - puv.y) < eps;

    u1 = pm[0]*signx1 + pm[3];
    u2 = pm[8]*signx1 + pm[11];

    u3 = pm[0]*signx2 + pm[3];
    u4 = pm[8]*signx2 + pm[11];

    // printf("P:(%f, %f)\n", px, py);

    for(int iy = y_start; iy < y_end; ++ iy, oy += dy) {
        idx = iy * nx + ix;

        signy1 = iy - 0.5f;
        signy2 = iy + 0.5f;

        if (!singular) {
            us[0] = (u1 + pm[1]*signy1) / (u2 + pm[9]*signy1);
            us[1] = (u1 + pm[1]*signy2) / (u2 + pm[9]*signy2);
            us[2] = (u3 + pm[1]*signy1) / (u4 + pm[9]*signy1);
            us[3] = (u3 + pm[1]*signy2) / (u4 + pm[9]*signy2);
        }
        else {
            us[0] = ((u1 + pm[1]*signy1)/(u2 + pm[9]*signy1)*1.5f - 1.0f) / puv.x;
            us[1] = ((u1 + pm[1]*signy2)/(u2 + pm[9]*signy2)*1.5f - 1.0f) / puv.x;
            us[2] = ((u3 + pm[1]*signy1)/(u4 + pm[9]*signy1)*1.5f - 1.0f) / puv.x;
            us[3] = ((u3 + pm[1]*signy2)/(u4 + pm[9]*signy2)*1.5f - 1.0f) / puv.x;
        }

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
            a0s = vecs[nu*2 + idxu];
            sb0 = vecs[nu*3 + idxu];
            lsp = vecs[nu*4 + idxu];

            r1 = ((ox-px)*(sx-px) + (oy-py)*(sy-py)) / (lsp*lsp);
            vblur = (a0s + sb0) * (-r1);
            y = crossg(sx-px, sy-py, ox-0.5f*dx-px, oy+0.5f*dy-py) / lsp;
            y += a0s * r1;

            // if (fabs(y) > fabs(vx) + fabs(vy) + fabs(vblur))
            //     continue;

            conv = linear_convolution_1d(-y, vx, vy, vblur);

            // printf("\t\t[U=%d] S:(%f, %f), P:(%f, %f), vx: %lf, vy: %lf, a0s: %lf, sb0: %lf, lsp: %lf, r1: %lf, vblur: %lf, y: %lf -> %lf [conv] %lf\n", idxu, sx, sy, px, py, vx, vy, a0s, sb0, lsp, r1, vblur, y-a0s*r1, y, conv);

            if(idxu < nu && idxu >= 0 && conv == conv && conv > eps)
                val += conv * proj[idxu] * dx * dy;
        }

        atomicAdd(vol+idx, val);
    }
}

__host__ static void generate_vectors(double *vecs, GeoData *geo) {
    int np = geo->np;
    int nu = geo->nuv.x;
    double dx = geo->dxyz.x;
    double dy = geo->dxyz.y;

    double cx, cy;
    double ux, uy;
    double px, py;
    double sx, sy, ax, ay, bx, by;
    double theta;
    double vx, vy, a0s, sb0, lsp;

    for(int p = 0; p < np; p++) {

        cx = *geo->dtvs[p*3], cy = *geo->dtvs[p*3+1];
        ux = *geo->puvs[p*3], uy = *geo->puvs[p*3+1];
        px = *geo->srcs[p*3], py = *geo->srcs[p*3+1];

        for(int u = 0; u < nu; u++) {
            sx = cx + ux * (u - 0.5 * nu + 0.5);
            sy = cy + uy * (u - 0.5 * nu + 0.5);
            ax = sx - ux * 0.5;
            ay = sy - uy * 0.5;
            bx = sx + ux * 0.5;
            by = sy + uy * 0.5;

            lsp = sqrt((px-sx)*(px-sx) + (py-sy)*(py-sy));

            // printf("%d, %d, %lf, %lf, %lf, %lf, %lf\n", p, u, px, py, sx, sy, lsp);

            theta = atan2(fabs(px-sx), fabs(py-sy));
            vx = fabs(dx * cos(theta)), vy = fabs(dy * sin(theta));

            if(py < sy) vx = -vx;
            if(px < sx) vy = -vy;

            a0s = fabs(cross(sx-px, sy-py, ax-px, ay-py)) / lsp;
            sb0 = fabs(cross(sx-px, sy-py, bx-px, by-py)) / lsp;

            a0s /= ((ax-px)*(sx-px) + (ay-py)*(sy-py)) / (lsp*lsp);
            sb0 /= ((bx-px)*(sx-px) + (by-py)*(sy-py)) / (lsp*lsp);

            vecs[p*nu*5 + nu*0 + u] = vx;
            vecs[p*nu*5 + nu*1 + u] = vy;
            vecs[p*nu*5 + nu*2 + u] = a0s;
            vecs[p*nu*5 + nu*3 + u] = sb0;
            vecs[p*nu*5 + nu*4 + u] = lsp;
        }
    }
}

A_B0::A_B0(GeoData *geo)
{
    geodata = geo;

    ngrid = 1;

    nblock = (geo->nuv.x + (ngrid-1)) / ngrid;

    vblock = dim3(16, 8);

    int bx = (geo->nxyz.x + vblock.x - 1) / vblock.x;
    int by = (geo->nxyz.y + (vgrid.y * Z_SIZE - 1)) / (vgrid.y * Z_SIZE);

    vgrid = dim3(bx, by);

    vecs = new MatrixD(geo->np * geo->nuv.x, 5);
    vecs->allocateMemory();

    generate_vectors((*vecs)[0], geo);
    vecs->copyHostToDevice();
}

A_B0::~A_B0()
{
    delete vecs;
}

void A_B0::project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
#endif
        b0_forward_projection<<<vgrid, vblock>>>(proj(int(p * geodata->nuv.x)), vol(0), (*vecs)(p * geodata->nuv.x * 5), geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3((float)geodata->dxyz.x, (float)geodata->dxyz.y, (float)geodata->dxyz.z), geodata->nuv.x,
                                                        make_float3((float)*geodata->srcs[p*3], (float)*geodata->srcs[p*3+1], (float)*geodata->srcs[p*3+2]),
                                                        make_float3((float)*geodata->puvs[p*3], (float)*geodata->puvs[p*3+1], (float)*geodata->puvs[p*3+2]),
                                                        make_float3((float)*geodata->dtvs[p*3], (float)*geodata->dtvs[p*3+1], (float)*geodata->dtvs[p*3+2]));
    }
}

void A_B0::back_project(MatrixD &vol, MatrixD &proj, double weight)
{
    for(int p=0; p<geodata->np; p++){

#ifdef DEBUG
        std::cout << p << std::endl;
#endif

        b0_backward_projection<<<vgrid, vblock>>>(proj(int(p * geodata->nuv.x)), vol(0), (*vecs)(p * geodata->nuv.x * 5), geodata->pmis(p*12), geodata->nxyz, 
                                                        make_float3((float)geodata->dxyz.x, (float)geodata->dxyz.y, (float)geodata->dxyz.z), geodata->nuv.x,
                                                        make_float3((float)*geodata->srcs[p*3], (float)*geodata->srcs[p*3+1], (float)*geodata->srcs[p*3+2]),
                                                        make_float3((float)*geodata->puvs[p*3], (float)*geodata->puvs[p*3+1], (float)*geodata->puvs[p*3+2]),
                                                        make_float3((float)*geodata->dtvs[p*3], (float)*geodata->dtvs[p*3+1], (float)*geodata->dtvs[p*3+2]));
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

            b0_matrix_generate<<<ngrid, nblock>>>(mat(0), geo->nxyz, geo->dxyz, geo->nuv.x,
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

    A_B0 *b0_init(int nx, int ny, int np, int nu, double dx, double dy, double du, double lsd, double lso, double *angles){
        GeoData *geo = new GeoData(nx, ny, 1, nu, 1, np, dx, dy, 1, du, 1);
        geo->geo_init_angles(lsd, lso, angles);
        geo->initialize_projection_matrix();

        A_B0 *b0_layer = new A_B0(geo);

        return b0_layer;
    }

    int b0_forward_projection(double *b, double *x, A_B0 *b0_layer) {
        int nx = b0_layer->geodata->nxyz.x;
        int ny = b0_layer->geodata->nxyz.x;

        int np = b0_layer->geodata->np;
        int nu = b0_layer->geodata->nuv.x;

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

        b0_layer->project(vol, proj, 1.0);

        cudaMemcpyFromSymbol(&cnt, conv_count, sizeof(int));
        // fprintf(stderr ,"Conv count: %d\n", cnt);

        cudaDeviceSynchronize();

        proj.copyDeviceToHost();

        memcpy(b, proj[0], np*nu*sizeof(double));

        return cnt;
    }

    void b0_backward_projection(double *b, double *x, A_B0 *b0_layer) {
        int nx = b0_layer->geodata->nxyz.x;
        int ny = b0_layer->geodata->nxyz.x;

        int np = b0_layer->geodata->np;
        int nu = b0_layer->geodata->nuv.x;

        MatrixD vol(nx, ny);
        MatrixD proj(np, nu);

        vol.allocateMemory();
        proj.allocateMemory();

        memset(vol[0], 0, nx*ny*sizeof(double));
        memcpy(proj[0], b, np*nu*sizeof(double));

        vol.copyHostToDevice();
        proj.copyHostToDevice();

        b0_layer->back_project(vol, proj, 1.0);

        // fprintf(stderr, "b, x, %p, %p, %lf", b, x, b[10000]);

        cudaDeviceSynchronize();

        vol.copyDeviceToHost();

        // printf("%p, %p, %p, %lf\n", vol[0], vol(0), x, *(vol[100]));

        for(int ix=0; ix<nx; ++ix)
            for(int iy=0; iy<ny; ++iy)
                x[ix*ny+iy] = *(vol[iy * nx + ix]);
    }
}


