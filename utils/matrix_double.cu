#include "matrix_double.hh"
#include "exception.hh"

MatrixD::MatrixD(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{ }

MatrixD::MatrixD(Shape shape) :
	MatrixD(shape.x, shape.y)
{ }

void MatrixD::allocateCudaMemory() {
	if (!device_allocated) {
		double* device_memory = nullptr;
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(double));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
		data_device = std::shared_ptr<double>(device_memory,
											 [&](double* ptr){ cudaFree(ptr); });
		device_allocated = true;
	}
}

void MatrixD::allocateHostMemory() {
	if (!host_allocated) {
		data_host = std::shared_ptr<double>(new double[shape.x * shape.y],
										   [&](double* ptr){ delete[] ptr; });
		host_allocated = true;
	}
}

void MatrixD::setCudaMemory(double *ptr) {
	if (!device_allocated) {
		data_device = std::shared_ptr<double>(ptr,
							[&](double* _){ });
		device_allocated = true;
	}
}

void MatrixD::setHostMemory(double *ptr) {
	if (!host_allocated) {
		data_host = std::shared_ptr<double>(ptr,
							[&](double* _){ });
		host_allocated = true;
	}
}

void MatrixD::allocateMemory() {
	allocateCudaMemory();
	allocateHostMemory();
}

void MatrixD::allocateMemoryIfNotAllocated(Shape shape) {
	if (!device_allocated && !host_allocated) {
		this->shape = shape;
		allocateMemory();
	}
}

void MatrixD::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(double), cudaMemcpyHostToDevice);
		//std::cout << cudaGetLastError() << std::endl;
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void MatrixD::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << cudaGetLastError() << std::endl;
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	}
	else {
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}

double* MatrixD::operator[](const int index) {
	return &data_host.get()[0] + index;
}

const double* MatrixD::operator[](const int index) const {
	return &data_host.get()[0] + index;
}

double* MatrixD::operator()(const int index) {
	return &data_device.get()[0] + index;
}

const double* MatrixD::operator()(const int index) const {
	return &data_device.get()[0] + index;
}
