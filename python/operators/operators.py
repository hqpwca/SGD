import numpy as np
import scipy
import ctypes
from ctypes import *
# import matplotlib.pyplot as plt
from numba import jit, njit

import pylops
from pylops import LinearOperator

class A_B0_operator(LinearOperator):
    def __init__(self, nx, ny, nnp, nu, dx, dy, du, lso, lsd, dtype=np.float64):

    def _matvec(self, x):

    def _rmatvec(self, x):
