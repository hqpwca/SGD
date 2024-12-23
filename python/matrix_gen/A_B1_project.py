import sys

sys.path.append('../python')

import ctypes
from ctypes import *
# import cupy as cp
import numpy as np
from scipy.integrate import *
from scipy.sparse import bsr_array, save_npz

from config import *

def get_matrix_cuda():
    dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)
    func = dll.bilinear_matrix_generation
    func.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, POINTER(c_double), c_int]
    func.restype = c_int
    return func

__get_matrix = get_matrix_cuda()

def get_matrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso, mat, bsize):
    mat_p = mat.ctypes.data_as(POINTER(c_double))

    return __get_matrix(nnp, nu, nx, ny, dx, dy, du, lsd, lso, mat_p, bsize)

import pickle

if __name__ == '__main__':

    # nx = 100
    # ny = 100
    # nnp = 2
    # nu = 855
    # lsd = 156.25
    # lso = 78.125
    # dx = 1
    # dy = 1
    # du = 0.78125

    if len(sys.argv) < 2:
        print("Usage: python3 [code] [filename]")
        exit(0)

    np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=230)

    bsize = 300000000

    spmat = np.zeros((bsize, 3), dtype=np.float64)

    num_data = get_matrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso, spmat, bsize)

    spmat = spmat.T

    print(num_data, spmat.shape)
    print(np.mean(spmat[2]), np.max(spmat[2]), file=sys.stderr)
    # print(np.mean(mat, axis=(1)))

    spmat = bsr_array((spmat[2, :num_data], spmat[:2, :num_data]), shape=(nnp*nu, nx*ny))
    save_npz('../matrixes/A_B1_' + sys.argv[1] + '.pkl', spmat, compressed=False)