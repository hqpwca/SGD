import sys
import ctypes
from ctypes import *
# import cupy as cp
import numpy as np
from scipy.integrate import *
from scipy.sparse import bsr_array

from config import *

import pickle

def get_hexmatrix_cuda():
    dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)
    func = dll.h0_matrix_generation
    func.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, POINTER(c_double), c_int]
    func.restype = c_int
    return func

__get_hexmatrix = get_hexmatrix_cuda()

def get_hexmatrix(nnp, nu, nl, nc, du, dt, lsd, lso, mat, bsize):
    mat_p = mat.ctypes.data_as(POINTER(c_double))

    return __get_hexmatrix(nnp, nu, nl, nc, du, dt, lsd, lso, mat_p, bsize)


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=230, suppress=True)

    if len(sys.argv) < 2:
        print("Usage: python3 [code] [filename]")
        exit(0)

    bsize = 300000000

    spmat = np.zeros((bsize, 3), dtype=np.float64)

    num_data = get_hexmatrix(nnp, nu, H0_nl, H0_nc, du, H0_dt, lsd, lso, spmat, bsize)

    spmat = spmat.T

    print(num_data, np.mean(spmat[2, :num_data]), np.max(spmat[2, :num_data]), file=sys.stderr)
    # print(np.mean(mat, axis=(1)))

    spmat = bsr_array((spmat[2, :num_data], spmat[:2, :num_data]), shape=(nnp*nu, H0_nl*H0_nc))

    # xx = (spmat.sum(axis=1).reshape(nnp,nu)*dt*dt)
    # print(xx, file=sys.stderr)
    
    pickle.dump(spmat, open('./matrixes/A_H0_' + sys.argv[1] + '.pkl', 'wb'))