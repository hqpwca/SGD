import numpy as np
import math

from numba import njit

@njit
def spline_conv(y, depth, vectors):
    if vectors.shape[0] == 0:
        assert depth >= 1
        if y > 0:
            return y**(depth-1)
        else:
            return 0
    else:
        return (spline_conv(y, depth+1, vectors[:-1]) - spline_conv(y-vectors[-1], depth+1, vectors[:-1])) / vectors[-1]