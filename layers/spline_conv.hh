#pragma once 

#include <cuda_runtime.h>

template <class T>
__device__ inline void sort2(T* a, T* b)
{
    if (*a > *b)
    {
        T tmp = *b;
        *b = *a;
        *a = tmp;
    }
}

template <class T>
__device__ inline void sort3(T* a, T* b, T* c)
{
    sort2<T>(b, c);
    sort2<T>(a, b);
    sort2<T>(b, c);
}

template <class T>
__device__ inline void sort4(T* a, T* b, T* c, T* d)
{

    sort2<T>(a,b);
    sort2<T>(c,d);
    sort2<T>(a,c);
    sort2<T>(b,d);
    sort2<T>(b,c);

    // sort3
    // sort2(b, c);
    // sort2(a, b);
    // sort2(b, c);

}

template <class T>
__device__ inline void sort2d(T* a, T* b)
{
    if (*a < *b)
    {
        T tmp = *b;
        *b = *a;
        *a = tmp;
    }
}

template <class T>
__device__ inline void sort3d(T* a, T* b, T* c)
{
    sort2d<T>(b, c);
    sort2d<T>(a, b);
    sort2d<T>(b, c);
}

// Single Precision

__device__ inline float base_function_2x(float y)
{
    if(y > 0) return y;
    else return 0;
}

__device__ inline float forward_difference_2x_2(float y, float h) {
    return (base_function_2x(y) - base_function_2x(y-h))/h;
}

__device__ inline float forward_difference_2x_1(float y, float h, float h2) {
    return (forward_difference_2x_2(y, h) - forward_difference_2x_2(y-h2, h))/h2;
}

__device__ inline float base_function_3x(float y)
{
    if(y > 0) return y*y;
    else return 0;
}

__device__ inline float forward_difference_3x_3(float y, float h) {
    return (base_function_3x(y) - base_function_3x(y-h))/h;
}

__device__ inline float forward_difference_3x_2(float y, float h, float h2) {
    return (forward_difference_3x_3(y, h) - forward_difference_3x_3(y-h2, h))/h2;
}

__device__ inline float forward_difference_3x_1(float y, float h, float h2, float h3) {
    return (forward_difference_3x_2(y, h, h2) - forward_difference_3x_2(y-h3, h, h2))/h3;
}

__device__ inline float base_function_4x(float y)
{
    if(y > 0) return y*y*y;
    else return 0;
}

__device__ inline float forward_difference_4x_4(float y, float h) {
    return (base_function_4x(y) - base_function_4x(y-h))/h;
}

__device__ inline float forward_difference_4x_3(float y, float h, float h2) {
    return (forward_difference_4x_4(y, h) - forward_difference_4x_4(y-h2, h))/h2;
}

__device__ inline float forward_difference_4x_2(float y, float h, float h2, float h3) {
    return (forward_difference_4x_3(y, h, h2) - forward_difference_4x_3(y-h3, h, h2))/h3;
}

__device__ inline float forward_difference_4x_1(float y, float h, float h2, float h3, float h4) {
    return (forward_difference_4x_2(y, h, h2, h3) - forward_difference_4x_2(y-h4, h, h2, h3))/h4;
}

__device__ inline float base_function_5x(float y)
{
    if(y > 0) return y*y*y*y;
    else return 0;
}

__device__ inline float forward_difference_5x_5(float y, float h) {
    return (base_function_5x(y) - base_function_5x(y-h))/h;
}

__device__ inline float forward_difference_5x_4(float y, float h, float h2) {
    return (forward_difference_5x_5(y, h) - forward_difference_5x_5(y-h2, h))/h2;
}

__device__ inline float forward_difference_5x_3(float y, float h, float h2, float h3) {
    return (forward_difference_5x_4(y, h, h2) - forward_difference_5x_4(y-h3, h, h2))/h3;
}

__device__ inline float forward_difference_5x_2(float y, float h, float h2, float h3, float h4) {
    return (forward_difference_5x_3(y, h, h2, h3) - forward_difference_5x_3(y-h4, h, h2, h3))/h4;
}

__device__ inline float forward_difference_5x_1(float y, float h, float h2, float h3, float h4, float h5) {
    return (forward_difference_5x_2(y, h, h2, h3, h4) - forward_difference_5x_2(y-h5, h, h2, h3, h4))/h5;
}

// __host__ inline float cross(float x1, float y1, float x2, float y2) {
//     return x1*y2 - x2*y1;
// }

__device__ inline float crossg(float x1, float y1, float x2, float y2) {
    return x1*y2 - x2*y1;
}

// Double Precision
__device__ inline double base_function_2x(double y)
{
    if(y > 0) return y;
    else return 0;
}

__device__ inline double forward_difference_2x_2(double y, double h) {
    return (base_function_2x(y) - base_function_2x(y-h))/h;
}

__device__ inline double forward_difference_2x_1(double y, double h, double h2) {
    return (forward_difference_2x_2(y, h) - forward_difference_2x_2(y-h2, h))/h2;
}

__device__ inline double base_function_3x(double y)
{
    if(y > 0) return y*y;
    else return 0;
}

__device__ inline double forward_difference_3x_3(double y, double h) {
    return (base_function_3x(y) - base_function_3x(y-h))/h;
}

__device__ inline double forward_difference_3x_2(double y, double h, double h2) {
    return (forward_difference_3x_3(y, h) - forward_difference_3x_3(y-h2, h))/h2;
}

__device__ inline double forward_difference_3x_1(double y, double h, double h2, double h3) {
    return (forward_difference_3x_2(y, h, h2) - forward_difference_3x_2(y-h3, h, h2))/h3;
}

__device__ inline double base_function_4x(double y)
{
    if(y > 0) return y*y*y;
    else return 0;
}

__device__ inline double forward_difference_4x_4(double y, double h) {
    return (base_function_4x(y) - base_function_4x(y-h))/h;
}

__device__ inline double forward_difference_4x_3(double y, double h, double h2) {
    return (forward_difference_4x_4(y, h) - forward_difference_4x_4(y-h2, h))/h2;
}

__device__ inline double forward_difference_4x_2(double y, double h, double h2, double h3) {
    return (forward_difference_4x_3(y, h, h2) - forward_difference_4x_3(y-h3, h, h2))/h3;
}

__device__ inline double forward_difference_4x_1(double y, double h, double h2, double h3, double h4) {
    return (forward_difference_4x_2(y, h, h2, h3) - forward_difference_4x_2(y-h4, h, h2, h3))/h4;
}

__device__ inline double base_function_5x(double y)
{
    if(y > 0) return y*y*y*y;
    else return 0;
}

__device__ inline double forward_difference_5x_5(double y, double h) {
    return (base_function_5x(y) - base_function_5x(y-h))/h;
}

__device__ inline double forward_difference_5x_4(double y, double h, double h2) {
    return (forward_difference_5x_5(y, h) - forward_difference_5x_5(y-h2, h))/h2;
}

__device__ inline double forward_difference_5x_3(double y, double h, double h2, double h3) {
    return (forward_difference_5x_4(y, h, h2) - forward_difference_5x_4(y-h3, h, h2))/h3;
}

__device__ inline double forward_difference_5x_2(double y, double h, double h2, double h3, double h4) {
    return (forward_difference_5x_3(y, h, h2, h3) - forward_difference_5x_3(y-h4, h, h2, h3))/h4;
}

__device__ inline double forward_difference_5x_1(double y, double h, double h2, double h3, double h4, double h5) {
    return (forward_difference_5x_2(y, h, h2, h3, h4) - forward_difference_5x_2(y-h5, h, h2, h3, h4))/h5;
}

__host__ inline double cross(double x1, double y1, double x2, double y2) {
    return x1*y2 - x2*y1;
}

__device__ inline double crossg(double x1, double y1, double x2, double y2) {
    return x1*y2 - x2*y1;
}

template <class T>
__device__ inline T univariate3dirboxspline_fast2(T x, T *xi) {
    return max(0.0f, min(1.0f / xi[0], ((xi[0] + xi[1]) / 2.0f - x) / (xi[0] * xi[1])));
}

template <class T>
__device__ inline T univariate3dirboxspline_fast3(T x, T *xi) {
    // Sort xi in descending order of absolute values

    // Compute normalization factors
    T h = 1.0f / xi[0]; // Using max |xi|
    T p = xi[0] * xi[1] * xi[2];

    // Compute the breakpoints K without taking absolute values of xi1, xi2, xi3
    T K[4];
    K[0] = (xi[0] - xi[1] - xi[2]) / 2.0f;
    K[1] = (xi[0] - xi[1] + xi[2]) / 2.0f;
    K[2] = (xi[0] + xi[1] - xi[2]) / 2.0f;
    K[3] = (xi[0] + xi[1] + xi[2]) / 2.0f;

    // printf("\t\t\tx:%f, xi:[%f, %f, %f], K:[%f, %f, %f, %f], h:%f, p:%f\n", x, xi[0], xi[1], xi[2], K[0], K[1], K[2], K[3], h, p);

    // Initialize output value
    T v;

    // bool cond1, cond2, cond3, cond4, cond5;

    // cond1 = x < K[3] & x >= K[2];
    // cond2 = x < K[2] & x >= K[1];
    // cond3 = x >= -K[0] & x < K[1];
    // cond4 = x < K[0];
    // cond5 = x < -K[0];

    // v = cond1?(0.5 * (K[3] - x) * (K[3] - x) / p):v;
    // v = cond2?(((xi[0] + xi[1]) / 2.0 - x) * xi[2] / p):v;
    // v = cond3?(h - 0.5 * (K[0] - x) * (K[0] - x) / p):v;
    // v = cond4?(h):v;
    // v = cond5?(h - (K[0]*K[0] + x*x) / p):v;

    if (x >= K[3]) {
        v = 0.0f;
    } else if (x >= K[2]) {
        v = 0.5f * (K[3] - x) * (K[3] - x) / p;
    } else if (x >= K[1]) {
        v = ((xi[0] + xi[1]) / 2.0f - x) * xi[2] / p;
    } else if (x < -K[0]) {
        v = h - (K[0]*K[0] + x*x) / p;
    } else if (x < K[0]) {
        v = h;
    } else {
        v = h - 0.5f * (K[0] - x) * (K[0] - x) / p;
    }

    return v;
}

template <class T>
__device__ inline T conv_2tri_1box_ag2b(T a, T b, T x) {
    T a3 = a*a*a;
    T a2b = a*a*b;
    T ab2 = a*b*b;
    T b3 = b*b*b;

    T k1 = (6*a*a*b*b);
    T k2 = a-b;
    T k3 = 3*a - b;

    T sum = 0.0f;

    if (x <= b) {
        return (x*(12*ab2 - 4*b3 - 4*b*x*x + x*x*x) ) / (2*k1);
    } else {
        sum += ((12*a - 7*b)*b) / (12*a*a);
    }

    if (x <= a-b) {
        return sum + (-2*a*b + b*b + 2*a*x - x*x) / (2*a*a);
    } else {
        sum += 0.5f - b/a;
    }

    if (x <= a) {
        return sum + ((a - b - x)*(a3 - 13*b3 + 9*b*b*x - 3*b*x*x - x*x*x - 3*a*a*(b + x) - 3*a*(b - x)*(3*b + x) )) / (4*k1);
    }
    else {
        sum += (13*b*b) / (24*a*a);
    }

    if (x < a+b) {
        return sum + ((x - a)*(a + 2*b - x)*(a*a + 2*a*b + 2*b*b - 2*(a + b)*x + x*x)) / (4*k1);
    }

    return sum + b*b / (24*a*a);
}

template <class T>
__device__ inline T conv_2tri_1box_al2b(T a, T b, T x) {
    T a3 = a*a*a;
    T a2b = a*a*b;
    T ab2 = a*b*b;
    T b3 = b*b*b;

    T k1 = (6*a*a*b*b);
    T k2 = a-b;
    T k3 = 3*a - b;

    T sum = 0.0f;

    if (x <= a-b) {
        return (x*(12*ab2 - 4*b3 - 4*b*x*x + x*x*x) ) / (2*k1);
    } else {
        sum += ((a - b)*(a3 - 7*a2b + 23*ab2 - 9*b3)) / (2*k1);
    }

    if (x <= b) {
        return sum + (((x - a + b)*(a3 - 11*a2b + 43*ab2 - 17*b3 + (5*a - 13*b)*k2*x - (a + 7*b)*x*x + 3*x*x*x)) / (4*k1));
    } else {
        sum += (((2*b - a)*(a3 - 6*a2b + 24*ab2 - 8*b3)) / (4*k1));
    }

    if (x <= a) {
        return sum + (((b - x)*(4*a3 - 3*b3 + b*b*x - 5*b*x*x - x*x*x - 6*a*a*(3*b + x) + 4*a*(b*b + 4*b*x + x*x))) / (4*k1));
    } else {
        sum += (((b - a)*(a3 - 7*a2b + 5*ab2 - 3*b3)) / (4*k1));
    }

    if (x < a+b) {
        return sum + ((x - a)*(a + 2*b - x)*(a*a + 2*a*b + 2*b*b - 2*(a + b)*x + x*x)) / (4*k1);
    }

    return sum + b*b / (24*a*a);
}

template <class T>
__device__ inline T fast_2tri_1box_spline(T a, T b, T x1, T x2) {
    T s1, s2;

    bool side = ((x1 * x2) > 0);
    x1 = abs(x1), x2 = abs(x2);

    if(b <= a-b) {
        s1 = conv_2tri_1box_ag2b<T>(a, b, x1);
        s2 = conv_2tri_1box_ag2b<T>(a, b, x2);
    }
    else {
        s1 = conv_2tri_1box_al2b<T>(a, b, x1);
        s2 = conv_2tri_1box_al2b<T>(a, b, x2);
    }

    T res = side?(abs(s1-s2)):(s1+s2);
    return res;
}


template <class T>
__device__ inline T fast_1tri_1box_spline(T a, T x1, T x2) {
    T s1, s2;

    bool side = ((x1 * x2) > 0);
    x1 = abs(x1), x2 = abs(x2);

    s1 = (x1>=a)?0.5f:(x1/a-(x1*x1)/(2*a*a));
    s2 = (x2>=a)?0.5f:(x2/a-(x2*x2)/(2*a*a));

    T res = side?(abs(s1-s2)):(s1+s2);
    return res;
}