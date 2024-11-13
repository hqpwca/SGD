#include <cuda_runtime.h>

__device__ inline double base_function_2x(double y)
{
    if(y > 0) return y;
    else return 0;
}

__device__ inline double forward_difference_2x_2(double y, double h) {
    double res;
    res = base_function_2x(y) - base_function_2x(y-h);
    return res/h;
}

__device__ inline double forward_difference_2x_1(double y, double h, double h2) {
    double res;
    res = forward_difference_2x_2(y, h) - forward_difference_2x_2(y-h2, h);
    return res/h2;
}

__device__ inline double base_function_3x(double y)
{
    if(y > 0) return y*y;
    else return 0;
}

__device__ inline double forward_difference_3x_3(double y, double h) {
    double res;
    res = base_function_3x(y) - base_function_3x(y-h);
    return res/h;
}

__device__ inline double forward_difference_3x_2(double y, double h, double h2) {
    double res;
    res = forward_difference_3x_3(y, h) - forward_difference_3x_3(y-h2, h);
    return res/h2;
}

__device__ inline double forward_difference_3x_1(double y, double h, double h2, double h3) {
    double res;
    res = forward_difference_3x_2(y, h, h2) - forward_difference_3x_2(y-h3, h, h2);
    return res/h3;
}

__device__ inline double base_function_4x(double y)
{
    if(y > 0) return y*y*y;
    else return 0;
}

__device__ inline double forward_difference_4x_4(double y, double h) {
    double res;
    res = base_function_4x(y) - base_function_4x(y-h);
    return res/h;
}

__device__ inline double forward_difference_4x_3(double y, double h, double h2) {
    double res;
    res = forward_difference_4x_4(y, h) - forward_difference_4x_4(y-h2, h);
    return res/h2;
}

__device__ inline double forward_difference_4x_2(double y, double h, double h2, double h3) {
    double res;
    res = forward_difference_4x_3(y, h, h2) - forward_difference_4x_3(y-h3, h, h2);
    return res/h3;
}

__device__ inline double forward_difference_4x_1(double y, double h, double h2, double h3, double h4) {
    double res;
    res = forward_difference_4x_2(y, h, h2, h3) - forward_difference_4x_2(y-h4, h, h2, h3);
    return res/h4;
}

__device__ inline double base_function_5x(double y)
{
    if(y > 0) return y*y*y*y;
    else return 0;
}

__device__ inline double forward_difference_5x_5(double y, double h) {
    double res;
    res = base_function_5x(y) - base_function_5x(y-h);
    return res/h;
}

__device__ inline double forward_difference_5x_4(double y, double h, double h2) {
    double res;
    res = forward_difference_5x_5(y, h) - forward_difference_5x_5(y-h2, h);
    return res/h2;
}

__device__ inline double forward_difference_5x_3(double y, double h, double h2, double h3) {
    double res;
    res = forward_difference_5x_4(y, h, h2) - forward_difference_5x_4(y-h3, h, h2);
    return res/h3;
}

__device__ inline double forward_difference_5x_2(double y, double h, double h2, double h3, double h4) {
    double res;
    res = forward_difference_5x_3(y, h, h2, h3) - forward_difference_5x_3(y-h4, h, h2, h3);
    return res/h4;
}

__device__ inline double forward_difference_5x_1(double y, double h, double h2, double h3, double h4, double h5) {
    double res;
    res = forward_difference_5x_2(y, h, h2, h3, h4) - forward_difference_5x_2(y-h5, h, h2, h3, h4);
    return res/h5;
}