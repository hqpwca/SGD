#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <exception>
#include <chrono>

#include "assert.h"
#include "stdlib.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "network.hh"
#include "layers/A_B1.hh"
#include "layers/Ref_projector.hh"
#include "utils/geo.hh"

#define BATCH_SIZE 256

int main() {
    GeoData *geo = new GeoData(100, 100, 1, 855, 1, 180, 1, 1, 0, 0.78125, 0);
    geo->geo_init_example(156.25, 78.125, 0.0f, PI*2 * 179/180);

    A_B1 *bl_layer = new A_B1(geo);
    MatrixD x(geo->nxyz.z * geo->nxyz.y * geo->nxyz.x, 1);
    x.allocateMemory();
    std::fill(x[0], x[geo->nxyz.z * geo->nxyz.y * geo->nxyz.x], 1.0f);
    x.copyHostToDevice();

    MatrixD y(geo->np * geo->nuv.x * geo->nuv.y, 1);
    y.allocateMemory();
    std::fill(y[0], y[geo->np * geo->nuv.x * geo->nuv.y], 0.0f);
    y.copyHostToDevice();

    std::cerr<< "Finished generating input data" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    bl_layer->project(x, y, 1.0);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    y.copyDeviceToHost();

    std::cerr<< "Finished Bilinear forward projection for 180 projs in " << duration.count() << " milliseconds." << std::endl;

    int nu = geo->nuv.x;
    int nv = geo->nuv.y;
    int np = geo->np;

    // for (int j=0; j<np; ++j) {
    //     double m1 = *std::max_element(y[j*nu], y[(j+1)*nu]);

    //     for (int i=0; i<geo->nuv.x; ++i) {
    //         std::cout << *y[geo->nuv.x*j + i] / m1 << ' ';
    //     }
    //     std::cout << std::endl;
    // }

    //delete geo;
    //delete geo2;
    
    return 0;
}

