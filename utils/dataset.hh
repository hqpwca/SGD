#pragma once

#include "matrix.hh"

#include <memory>

enum select_mode {RandomPermute, RandomChoose};

class Dataset {
    std::vector<float *> inputs;
    std::vector<float *> outputs;
    std::vector<int> ind;
    std::vector<float> T;

    Matrix input; //(num_input, batch_size) !column_major
    Matrix output; //(num_output, batch_size) !column_major

    size_t batch_size, num_data;
    select_mode mode;

    int cur_epoch;
    int cur_bnum;

    std::default_random_engine generator;
public:
    Dataset(std::vector<float> &T, int num_input = 16, int num_output = 1, int batch_size = 32, int num_data = 2048, select_mode mode = RandomPermute);
    ~Dataset();

    Matrix getBatchInput();
    Matrix getBatchOutput();
    bool nextBatch();
    int nextEpoch();
};