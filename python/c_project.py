import sys
import ctypes
from ctypes import *
# import cupy as cp
import numpy as np
from scipy.integrate import *

from numba import jit

np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=10000)

def get_matrix_cuda():
    dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)
    func = dll.bilinear_matrix_generation
    func.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, POINTER(c_double)]
    return func

__get_matrix = get_matrix_cuda()

def get_matrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso, mat):
    mat_p = mat.ctypes.data_as(POINTER(c_double))

    __get_matrix(nnp, nu, nx, ny, dx, dy, du, lsd, lso, mat_p)

if __name__ == '__main__':
    nx = 10
    ny = 10
    nnp = 16
    nu = 128
    lsd = 11.0
    lso = 5.0
    dx = 0.1
    dy = 0.1
    du = 0.025

    mat = np.zeros((nnp, nu, nx, ny), dtype=np.float64)

    get_matrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso, mat)

    px = np.sum(mat, axis=(0, 1))

    print(px)