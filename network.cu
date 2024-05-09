#include "network.hh"

Network::Network(float lr) :
	lr(lr)
{ }

Network::~Network() {
	for (auto layer : layers) {
		delete layer;
	}
}

void Network::addLayer(Layer* layer) {
	this->layers.push_back(layer);
}

Matrix& Network::forward(cublasHandle_t &cublasH, Matrix &x) {
	Matrix tmp = x;

	for (auto layer : layers) {
		tmp = layer->forward(cublasH, tmp);
	}

	output = tmp;
	return output;
}

void Network::back_prop(cublasHandle_t &cublasH, Matrix &loss, float lr) {
    for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		loss = (*it)->back_prop(cublasH, loss, lr);
	}
    cudaDeviceSynchronize();
}

std::vector<Layer*> Network::getLayers() const {
	return layers;
}
