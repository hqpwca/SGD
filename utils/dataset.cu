#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>

#include "dataset.hh"

// Target function: Y = F(X) = (x1^t1 + ((x1+x2)/2)^t2 + ... + ((x1+...+x16)/16)^t16)/16 (t_i \in [0.5, 2]) (x_i \in [0.1, 1])

Dataset::Dataset(std::vector<float> &T, int num_input, int num_output, int batch_size, int num_data, select_mode mode)
    : T(T), input(num_input, batch_size), output(num_output, batch_size), batch_size(batch_size), num_data(num_data), mode(mode)
{
    cur_epoch = -1;
    cur_bnum = -1;

    std::uniform_real_distribution<float> rand_x(0.1, 1.0);
    std::uniform_real_distribution<float> rand_t(0.5, 2.0);

    generator.seed(time(NULL));

    if (T.size() != num_input) {
        T.clear();
        for(int i = 0; i < num_input; ++i)
            T.push_back(rand_t(generator));
    }

    for(int i = 0; i < num_data; ++i) {
        float *X = new float[num_input];
        float *Y = new float[num_output];
        float sumX = 0.0;

        std::fill(Y, Y+num_output, 0.0);
        for(int idx = 0; idx < num_input; ++idx) {
            X[idx] = rand_x(generator);
            sumX += X[idx];
            
            Y[0] += std::pow(sumX/(idx+1), T[idx]);
        }
        Y[0] /= num_input;

        inputs.push_back(X);
        outputs.push_back(Y);
        ind.push_back(i);
    }

    input.allocateMemory();
    output.allocateMemory();
}

Dataset::~Dataset() {
    for(auto i : inputs)
        delete[] i;
    for(auto i : outputs)
        delete[] i;
}

Matrix Dataset::getBatchInput() {
    assert(cur_epoch >= 0 && cur_bnum >= 0);

    input.copyHostToDevice();
    return input;
}

Matrix Dataset::getBatchOutput() {
    assert(cur_epoch >= 0 && cur_bnum >= 0);

    output.copyHostToDevice();
    return output;
}

bool Dataset::nextBatch() {
    if((++ cur_bnum) * batch_size + batch_size > num_data)
        return false;

    for (int i = 0; i < batch_size; ++i) {
        int idx = ind[cur_bnum * batch_size + i];
        
        // TODO: change memcpy to setup of pointers.
        memcpy(input[i*input.shape.x], inputs[idx], sizeof(float) * input.shape.x);
        memcpy(output[i*output.shape.x], outputs[idx], sizeof(float) * output.shape.x);
    }

    return true;
}

int Dataset::nextEpoch() {
    std::shuffle(ind.begin(), ind.end(), generator);
 
    cur_epoch ++;
    cur_bnum = -1;
}