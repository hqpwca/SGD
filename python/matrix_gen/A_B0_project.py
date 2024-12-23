import sys

sys.path.insert(0, '/home/ke/SGD/python')

import ctypes
from ctypes import *
# import cupy as cp
import numpy as np
from scipy.integrate import *
from scipy.sparse import bsr_array, save_npz

from config import *

def get_pmatrix_cuda():
    dll = CDLL('../../build/libnetwork.so', mode = RTLD_GLOBAL)
    func = dll.pixel_matrix_generation
    func.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, POINTER(c_double), c_int]
    func.restype = c_int
    return func

__get_pmatrix = get_pmatrix_cuda()

def get_pmatrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso, mat, bsize):
    mat_p = mat.ctypes.data_as(POINTER(c_double))

    return __get_pmatrix(nnp, nu, nx, ny, dx, dy, du, lsd, lso, mat_p, bsize)

import pickle

if __name__ == '__main__':
    nx = 1
    ny = 1
    nnp = 2
    nu = 5
    lsd = 10
    lso = 5
    dx = 1
    dy = 1
    du = 1

    if len(sys.argv) < 2:
        print("Usage: python3 [code] [filename]")
        exit(0)

    np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=230)

    bsize = 300000000

    spmat = np.zeros((bsize, 3), dtype=np.float64)

    num_data = get_pmatrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso, spmat, bsize)

    spmat = spmat.T

    print(num_data, spmat.shape)
    print(np.mean(spmat[2]), np.max(spmat[2]), file=sys.stderr)
    # print(np.mean(mat, axis=(1)))

    spmat = bsr_array((spmat[2, :num_data], spmat[:2, :num_data]), shape=(nnp*nu, nx*ny))
    if sys.argv[1] != 'test':
        save_npz('../matrixes/A_B0_' + sys.argv[1] + '.pkl', spmat, compressed=False)