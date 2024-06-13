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

#include "assert.h"
#include "stdlib.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "network.hh"
#include "layers/SF_projector.hh"
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
    GeoData *geo = new GeoData(256, 256, 256, 505, 523, 256, 0.15, 0.15, 0.15, 0.2, 0.2);
    geo->geo_init_example(800, 600, 0.0f, 255.0*PI/128.0);
    geo->initialize_projection_matrix();

    SF *sf_layer = new SF(geo);

    Matrix x(256*256*256, 1);
    x.allocateMemory();

    for(int i=0; i<256; ++i){ //z
        for(int j=0; j<256; ++j){ //y
            for(int k=0; k<256; ++k) { //x
                int idx = i*256*256+j*256+k;
                double di = i-128, dj = j-128, dk = k-128;
                if(sqrt(di*di + dj*dj + dk*dk) < 100.0){
                    *x[idx] = 1.0f;
                }
                else {
                    *x[idx] = 0.0f;
                }
            }
        }
    }
    x.copyHostToDevice();

    Matrix y(256*510*525, 1);
    y.allocateMemory();
    std::fill(y[0], y[256*510*525-1], 0.0f);
    y.copyHostToDevice();

    std::cout<< "Finished generating input data" << std::endl;

    sf_layer->project(x, y, 1.0);

    std::cout<< "Finished forward projection" << std::endl;

    cudaDeviceSynchronize();

    y.copyDeviceToHost();

    std::cout<< "Finished copy to host" << std::endl;

    std::cout << *y[100*505*523+505*200+300] << std::endl;
    
    return 0;
}