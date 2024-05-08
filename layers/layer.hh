#pragma once

#include <iostream>

#include "../utils/matrix.hh"

enum Ltype {TLinear, TReLU, TSigmoid};

class Layer {
public:
    Ltype type;

	virtual ~Layer() = 0;

	virtual Matrix& forward(cublasHandle_t &cublasH, Matrix &X) = 0;
	virtual Matrix& back_prop(cublasHandle_t &cublasH, Matrix &od, float lr) = 0;
};

inline Layer::~Layer() {}