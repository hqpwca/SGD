#pragma once

#include "shape.hh"

#include <memory>

class MatrixD {
private:
	bool device_allocated;
	bool host_allocated;

public:
	Shape shape;

	std::shared_ptr<double> data_device;
	std::shared_ptr<double> data_host;

	MatrixD(size_t x_dim = 1, size_t y_dim = 1);
	MatrixD(Shape shape);

	void allocateMemory();
	void allocateCudaMemory();
	void allocateHostMemory();
	void allocateMemoryIfNotAllocated(Shape shape);

	void setCudaMemory(double *ptr);
	void setHostMemory(double *ptr);

	void copyHostToDevice();
	void copyDeviceToHost();

	double* operator[](const int index);
	const double* operator[](const int index) const;

	double* operator()(const int index);
	const double* operator()(const int index) const;
};
