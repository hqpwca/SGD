/*
    A program trying to implement SGD using CUDA.
    Target function: Y = F(X) = (x1^t1 + ((x1+x2)/2)^t2 + ... + ((x1+...+x16)/16)^t16)/16 (t_i \in [0.5, 2]) (x_i \in [0, 1])
    Network Structure: Input -- Linear(16, 32) -- ReLU -- Linear(32, 16) -- ReLU -- Linear(16, 1) -- Sigmoid
*/

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
#include "layers/SF_projector.hh"
#include "layers/Ref_projector.hh"
#include "utils/dataset.hh"

#define BATCH_SIZE 256

cublasHandle_t handle;
std::vector<float> T;
Dataset *train, *val, *test;

Matrix calcSquareLoss(Matrix &batch_output, Matrix &network_output) // (1, batch_size)
{
    assert(batch_output.shape.x == network_output.shape.x && batch_output.shape.y == network_output.shape.y);

    Matrix m(batch_output.shape);
    m.allocateMemory();
    for (int i = 0; i < batch_output.shape.x * batch_output.shape.y; ++ i) {
        *m[i] =  std::pow((network_output[i] - batch_output[i]), 2.0);
    }
    return m;
}

float sumLoss(Matrix &m) {
    float sum = 0.0;
    for (int i = 0; i < m.shape.x * m.shape.y; ++ i) {
        sum += *m[i];
    }
    return sum;
}

void init_network(Network &N) {
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    N.addLayer(new Linear(Shape(16, 16)));
    N.addLayer(new ReLU());
    N.addLayer(new Linear(Shape(16, 32)));
    N.addLayer(new ReLU());
    N.addLayer(new Linear(Shape(32, 16)));
    N.addLayer(new Sigmoid());
    N.addLayer(new Linear(Shape(16, 1)));
}

void cleanup() {
    cublasDestroy(handle);
    delete train;
    delete test;
}

void train_SGD(Network &N, Dataset *train, Dataset *val, float lr = 0.1, int epoches = 100, int batch_size = BATCH_SIZE) {
    for (int epoch = 0; epoch < epoches; ++epoch) {
        train->nextEpoch();

        int num_batches = -1;
        float sumloss = 0.0;
        while(train->nextBatch()) {
            ++ num_batches;
            Matrix input = train->getBatchInput();
            Matrix output = train->getBatchOutput();

            //Forward
            Matrix net_output = N.forward(handle, input);
            net_output.copyDeviceToHost();

            //Calc loss
            Matrix loss = calcSquareLoss(output, net_output);
            sumloss += sumLoss(loss);
            loss.copyHostToDevice();

            //Back prop
            N.back_prop(handle, loss, lr*10/(10+epoch)); //Dynamic Learning Rate
        }

        std::cerr << "Finished Epoch #" << epoch+1 << " Train Loss: " << sumloss / num_batches / batch_size;

        //Validate
        val->nextEpoch();

        num_batches = -1;
        float sumval = 0.0;
        while(val->nextBatch()) {
            ++ num_batches;
            Matrix input = val->getBatchInput();
            Matrix output = val->getBatchOutput();

            //Forward
            Matrix net_output = N.forward(handle, input);
            net_output.copyDeviceToHost();

            //Calc loss
            Matrix loss = calcSquareLoss(output, net_output);
            sumval += sumLoss(loss);
        }
        std::cerr << " Valid Loss: " << sumval / num_batches / batch_size << std::endl;
    }
}

void validate_SGD(Network &N, Dataset *d) {

}

void test_SGD(Network &N, Dataset *d) {
    d->nextEpoch();

    int num_batches = -1;
    float sumloss = 0.0;
    while(d->nextBatch()) {
        ++ num_batches;
        Matrix input = d->getBatchInput();
        Matrix output = d->getBatchOutput();

        //Forward
        Matrix net_output = N.forward(handle, input);
        net_output.copyDeviceToHost();

        //Calc loss
        Matrix loss = calcSquareLoss(output, net_output);
        sumloss += sumLoss(loss);
    }

    std::cerr << "Finished Epoch #" << "Test" << " Loss: " << sumloss / num_batches / BATCH_SIZE << std::endl;
}

/*
int main(int argc, char *argv[]) {
    Network N;
    init_network(N);

    train = new Dataset(T, 16, 1, BATCH_SIZE, 4096);
    val = new Dataset(T, 16, 1, BATCH_SIZE, 512);
    test = new Dataset(T, 16, 1, BATCH_SIZE, 512);

    train_SGD(N, train, val, 0.05);

    test_SGD(N, test);

    cleanup();

    return 0;
}
*/

int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, 12928);
    GeoData *geo = new GeoData(128, 128, 1, 256, 1, 360, 1, 1, 0, 0.75, 0);
    //GeoData *geo = new GeoData(100, 100, 100, 500, 500, 360, 0.01, 0.01, 0.01, 0.01, 0.01);
    //GeoData *geo = new GeoData(256, 256, 256, 500, 500, 360, 0.01, 0.01, 0.01, 0.0256, 0.0256);
    //GeoData *geo = new GeoData(400, 400, 160, 520, 264, 256, 0.15, 0.15, 0.15, 0.2, 0.2);
    geo->geo_init_example(11, 5, 0.0f, PI*2 * 359/360);
    //geo->geo_init_example(800, 600, 0.0f, PI);
    geo->initialize_projection_matrix();

    SF *sf_layer = new SF(geo);
    Ref *ref_layer = new Ref(geo);

    Matrix x(geo->nxyz.z * geo->nxyz.y * geo->nxyz.x, 1);
    x.allocateMemory();

    std::fill(x[0], x[geo->nxyz.z * geo->nxyz.y * geo->nxyz.x], 1.0f);

    // for(int i=0; i<geo->nxyz.z; ++i){ //z
    //     for(int j=0; j<geo->nxyz.y; ++j){ //y
    //         for(int k=0; k<geo->nxyz.x; ++k) { //x
    //             int idx = i*geo->nxyz.y*geo->nxyz.x + j*geo->nxyz.x + k;
    //             double di = i-(geo->nxyz.z/2), dj = j-(geo->nxyz.y/2), dk = k-(geo->nxyz.x/2);
    //             if(sqrt(di*di + dj*dj + dk*dk) < 1e-5){
    //                 *x[idx] = 1.0f;
    //             }
    //             else {
    //                 *x[idx] = 0.0f;
    //             }
    //         }
    //     }
    // }

    // Add a small empty sphere inside

    /*
    for(int i=0; i<geo->nxyz.z; ++i){ //z
        for(int j=0; j<geo->nxyz.y; ++j){ //y
            for(int k=0; k<geo->nxyz.x; ++k) { //x
                int idx = i*geo->nxyz.y*geo->nxyz.x + j*geo->nxyz.x + k;
                double di = i-80, dj = j-80, dk = k-80;
                if(sqrt(di*di + dj*dj + dk*dk) < 20.0){
                    *x[idx] = 0.0f;
                }
            }
        }
    }

    */

    x.copyHostToDevice();

    //Matrix y(256*510*525, 1);
    MatrixD y(geo->np * geo->nuv.x * geo->nuv.y, 1);
    y.allocateMemory();
    std::fill(y[0], y[geo->np * geo->nuv.x * geo->nuv.y], 0.0f);
    y.copyHostToDevice();

    std::cerr<< "Finished generating input data" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    //sf_layer->project(x, y, 1.0, true);
    ref_layer->project(x, y, 1.0);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cerr<< "Finished Ref forward projection for 4 projs in " << duration.count() << " milliseconds." << std::endl;

    MatrixD sy(geo->np * geo->nuv.x * geo->nuv.y, 1);
    sy.allocateMemory();
    std::fill(sy[0], sy[geo->np * geo->nuv.x * geo->nuv.y], 0.0f);
    sy.copyHostToDevice();

    start = std::chrono::high_resolution_clock::now();

    sf_layer->project(x, sy, 1.0, true);
    //ref_layer->project(x, y, 1.0);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cerr<< "Finished SF forward projection for 4 projs in " << duration.count() << " milliseconds." << std::endl;

    // float maxerr45 = 0.0f, maxerr22dot5 = 0.0f;

    // float max0 = *std::max_element(y[0 * geo->nuv.x * geo->nuv.y], y[(1) * geo->nuv.x * geo->nuv.y]);
    // float max45 = *std::max_element(y[32 * geo->nuv.x * geo->nuv.y], y[(33) * geo->nuv.x * geo->nuv.y]);
    // //float max225 = *std::max_element(y[16 * geo->nuv.x * geo->nuv.y], y[(17) * geo->nuv.x * geo->nuv.y]);

    // for(int u=0; u<geo->nuv.y; ++u) {
    //     for(int v=0; v<geo->nuv.x; ++v) {
    //         int cu = geo->nuv.y/2 - u;
    //         int cv = geo->nuv.x/2 - v;
    //         if(cu*cu+cv*cv < 5625) {
    //             maxerr45 = fmaxf(fabs(*y[u*geo->nuv.x + v]/max0- *y[32*geo->nuv.x*geo->nuv.y + u*geo->nuv.x + v]/max45), maxerr45);
    //             maxerr22dot5 = fmaxf(fabs(*y[u*geo->nuv.x + v]/max0 - *y[16*geo->nuv.x*geo->nuv.y + u*geo->nuv.x + v]/max225), maxerr22dot5);
    //         }
    //     }
    // }

    // std::cerr<< maxerr45 << std::endl;
    // std::cerr<< maxerr22dot5 << std::endl;

    // Matrix rx(geo->nxyz.z * geo->nxyz.y * geo->nxyz.x, 1);
    // rx.allocateMemory();
    // std::fill(rx[0], rx[geo->nxyz.z * geo->nxyz.y * geo->nxyz.x], 0.0f);
    // rx.copyHostToDevice();

    // std::cerr<< "Starting BackProjection" << std::endl;

    // start = std::chrono::high_resolution_clock::now();

    // sf_layer->back_project(rx, y, 1.0);

    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // std::cerr<< "Finished backward projection in " << duration.count() << " milliseconds." << std::endl;

    // rx.copyDeviceToHost();

    // std::cerr<< *std::max_element(rx[0], rx[geo->nxyz.z * geo->nxyz.y * geo->nxyz.x]) << std::endl;

    int nu = geo->nuv.x;
    int nv = geo->nuv.y;
    int np = geo->np;

    for(int p=0; p<np; ++p) {
        if(p != 0 && p != 15 && p != 35 && p != 45) continue;

        char ref[100], sf[100], dif[100], sft[100];
        sprintf(ref, "images/ref%03d.pgm", p);
        sprintf(sf, "images/sf%03d.pgm", p);
        sprintf(dif, "images/diff%03d.pgm", p);
        sprintf(sft, "images/sft%03d.pgm", p);
        FILE *fref = fopen(ref, "wb");
        FILE *fsf = fopen(sf, "wb");
        FILE *fdiff = fopen(dif, "wb");
        FILE *fsft = fopen(sft, "wb");
        fprintf(fref, "P5\n%i %i 255\n", nu-49, nv-49);
        fprintf(fsf, "P5\n%i %i 255\n", nu-49, nv-49);
        fprintf(fdiff, "P5\n%i %i 255\n", nu-49, nv-49);
        fprintf(fsft, "P5\n%i %i 255\n", nu, nv);
        // fprintf(f, "P5\n%i %i 255\n", nu, nv);

        double *sumy = new double[(nu+1)*(nv+1)];
        double *avgy = new double[nu*nv];
        memset(sumy, 0, nu*nv);
        memset(avgy, 0, nu*nv);

        double *sumsy = new double[(nu+1)*(nv+1)];
        double *avgsy = new double[nu*nv];
        memset(sumsy, 0, nu*nv);
        memset(avgsy, 0, nu*nv);

        for(int u=0; u<nu; ++u) {
            for(int v=0; v<nv; ++v) {
                sumy[(u+1)*nv + (v+1)] = *y[p*nu*nv+u*nv+v] + sumy[u*nv+(v+1)] + sumy[(u+1)*nv+v] - sumy[u*nv+v];
            }
        }

        for(int u=0; u<nu; ++u) {
            for(int v=0; v<nv; ++v) { // [u-49, u] x [v-49, v]
                int idx = u*nv + v;
                int idxtl = (u-49)*nv + (v-49);
                int idxbr = (u+1) *nv + (v+1) ;
                int idxbl = (u+1) *nv + (v-49);
                int idxtr = (u-49)*nv + (v+1) ;

                if (u < 49 || v < 49 || u >= nu || v >= nv) continue;
                else {
                    avgy[idx] = (sumy[idxbr] + sumy[idxtl] - sumy[idxbl] - sumy[idxtr]) / 2500.0;
                }
            }
        }

        for(int u=0; u<nu; ++u) {
            for(int v=0; v<nv; ++v) {
                sumsy[(u+1)*nv + (v+1)] = *sy[p*nu*nv+u*nv+v] + sumsy[u*nv+(v+1)] + sumsy[(u+1)*nv+v] - sumsy[u*nv+v];
            }
        }

        for(int u=0; u<nu; ++u) {
            for(int v=0; v<nv; ++v) { // [u-49, u] x [v-49, v]
                int idx = u*nv + v;
                int idxtl = (u-49)*nv + (v-49);
                int idxbr = (u+1) *nv + (v+1) ;
                int idxbl = (u+1) *nv + (v-49);
                int idxtr = (u-49)*nv + (v+1) ;

                if (u < 49 || v < 49 || u >= nu || v >= nv) continue;
                else {
                    avgsy[idx] = (sumsy[idxbr] + sumsy[idxtl] - sumsy[idxbl] - sumsy[idxtr]) / 2500.0;
                }
            }
        }

        float max = *std::max_element(avgy, avgy + nv * nu);
        float maxs = *std::max_element(avgsy, avgsy + nv * nu);

        float *diff = new float[nu * nv];
        memset(diff, 0, nu*nv);

        for(int u=49; u<nu; ++u){
            for(int v=49; v<nv; ++v){
                int idx = u*nv + v;

                diff[idx] = fabs(avgy[idx]/max - avgsy[idx]/maxs);
            }
        }
        
        float maxdiff = *std::max_element(diff, diff + nv * nu);

        std::cout << max << ' ' << maxs << ' ' << maxdiff << std::endl;

        for(int u=49; u<nu; ++u) {
            for(int v=49; v<nv; ++v) {
                //int idx = p*nv*nu + u*nv + v;
                int idx = u*nv + v;

                float val = avgy[idx];
                val *= 255.0f / max;
                val = fmaxf(0.0f, val);
                //unsigned char c = 255 - val;
                unsigned char c = val;
                fputc(c, fref);

                val = avgsy[idx];
                val *= 255.0f / maxs;
                val = fmaxf(0.0f, val);
                //unsigned char c = 255 - val;
                c = val;
                fputc(c, fsf);

                val = diff[idx];
                val *= 255.0f / 0.03;
                val = fmaxf(0.0f, val);
                val = fminf(255.0f, val);
                //unsigned char c = 255 - val;
                c = val;
                fputc(c, fdiff);
            }
        }

        max = *std::max_element(sy[p * geo->nuv.x * geo->nuv.y], sy[(p+1) * geo->nuv.x * geo->nuv.y]);

        for(int u=0; u<nu; ++u) {
            for(int v=0; v<nv; ++v) {
                int idx = p*nv*nu + u*nv + v;
                //int idx = p* u*nv + v;

                float val = *sy[idx];
                val *= 255.0f / max;
                val = fmaxf(0.0f, val);
                //unsigned char c = 255 - val;
                unsigned char c = val;
                fputc(c, fsft);
            }
        }
        fclose(fref);
        fclose(fsf);
        fclose(fdiff);
        fclose(fsft);

        delete[] sumy;
        delete[] avgy;
        delete[] sumsy;
        delete[] avgsy;
        delete[] diff;
    }

    // for(int z=0; z<geo->nxyz.z; ++z) {
    //     char filename[100];
    //     sprintf(filename, "back_images/backprojection%03d.pgm", z);
    //     FILE *f = fopen(filename, "wb");
    //     fprintf(f, "P5\n%i %i 255\n", geo->nxyz.y, geo->nxyz.x);

    //     float max = *std::max_element(rx[z * geo->nxyz.y * geo->nxyz.x], rx[(z+1) * geo->nxyz.y * geo->nxyz.x]);

    //     for(int u=0; u<geo->nxyz.y; ++u) {
    //         for(int v=0; v<geo->nxyz.x; ++v) {
    //             float val = *rx[z*geo->nxyz.x*geo->nxyz.y + u*geo->nxyz.x + v];
    //             val *= 255.0f / max;
    //             val = fmaxf(0.0f, val);
    //             //unsigned char c = 255 - val;
    //             unsigned char c = val;
    //             fputc(c, f);
    //         }
    //     }
    //     fclose(f);
    // }

    // delete sf_layer;
    delete ref_layer;
  
    return 0;
}