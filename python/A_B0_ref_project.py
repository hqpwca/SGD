import sys
import ctypes
# import cupy as cp
import numpy as np
import scipy
from scipy.integrate import *
from scipy.sparse import *

import numba as nb
from numba import jit, cfunc, carray

from config import *

import pickle

np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=10000)

c_sig = nb.types.double(nb.types.int32, nb.types.CPointer(nb.types.double))

@cfunc(c_sig)
def line_in_square(n, args):
    eps = 1e-9

    arg = carray(args, (n,), dtype = np.double)

    x, px, py, _sx, _sy, ux, uy, xmin, ymin, xmax, ymax = arg

    sx = _sx + x*ux
    sy = _sy + x*uy

    if np.fabs(px - sx) < eps:
        if px > xmin - eps and px < xmax + eps:
            return ymax - ymin
        else:
            return (0.0)
    
    if np.fabs(py - sy) < eps:
        if py > ymin - eps and py < ymax + eps:
            return (xmax - xmin)
        else:
            return (0.0)

    k = np.divide((py - sy) , (px - sx))
    
    d = (xmax - xmin) * np.sqrt(1.0 + k*k)

    i1 = sy + k * (xmin - sx)
    i2 = sy + k * (xmax - sx)

    if i2 < i1: 
        i1, i2 = i2, i1

    if i2 < (ymin - eps):
        return (0.0)
    
    if i1 > (ymax + eps):
        return (0.0)
    
    r = np.divide((min(i2, ymax) - max(i1, ymin)) , (i2 - i1))

    return d * r

@cfunc(c_sig)
def line_in_square_bl(n, args):
    eps = 1e-9

    arg = carray(args, (n,), dtype = np.double)

    x, px, py, _sx, _sy, ux, uy, xmin, ymin, xmax, ymax = arg

    sx = _sx + x*ux
    sy = _sy + x*uy

    if np.fabs(px - sx) < eps:
        if px > xmin - eps and px < xmax + eps:
            return np.exp(ymax - ymin)
        else:
            return np.exp(0.0)
    
    if np.fabs(py - sy) < eps:
        if py > ymin - eps and py < ymax + eps:
            return np.exp(xmax - xmin)
        else:
            return np.exp(0.0)

    k = np.divide((py - sy) , (px - sx))
    
    d = (xmax - xmin) * np.sqrt(1.0 + k*k)

    i1 = sy + k * (xmin - sx)
    i2 = sy + k * (xmax - sx)

    if i2 < i1: 
        i1, i2 = i2, i1

    if i2 < (ymin - eps):
        return np.exp(0.0)
    
    if i1 > (ymax + eps):
        return np.exp(0.0)
    
    r = np.divide((min(i2, ymax) - max(i1, ymin)) , (i2 - i1))

    #return d * r
    return np.exp(d * r)

# @jit(nopython = True)
# def target_function(x, px, py, sx, sy, ux, uy, xmin, ymin, xmax, ymax):
#     return line_in_square(px, py, sx+x*ux, sy+x*uy, xmin, ymin, xmax, ymax)

# @jit(nopython = True)
# def target_function_bl(x, px, py, sx, sy, ux, uy, xmin, ymin, xmax, ymax):
#     return np.power(np.e, line_in_square(px, py, sx+x*ux, sy+x*uy, xmin, ymin, xmax, ymax))

@jit(nopython = True)
def rot_sort4(v1, v2, v3, v4):
    if v1[0] * v2[1] < v1[1] * v2[0]:
        v1, v2 = v2, v1
    if v1[0] * v3[1] < v1[1] * v3[0]:
        v1, v3 = v3, v1
    if v1[0] * v4[1] < v1[1] * v4[0]:
        v1, v4 = v4, v1
    if v2[0] * v3[1] < v2[1] * v3[0]:
        v2, v3 = v3, v2
    if v2[0] * v4[1] < v2[1] * v4[0]:
        v2, v4 = v4, v2
    if v3[0] * v4[1] < v3[1] * v4[0]:
        v3, v4 = v4, v3
    
    return v1, v2, v3, v4

thread_list = []

def projection_matrix(nx, ny, nnp, nu, dx, dy, srcs, dtvs, nuvs, beers_law = False):
    points = []

    for x in range(nx):
        xmin = (x - (nx/2)) * dx
        xmax = xmin + dx

        for y in range(ny):
            ymin = (y - (ny/2)) * dy
            ymax = ymin + dy

            for p in range(nnp):
                px = srcs[p][0]
                py = srcs[p][1]

                ux = nuvs[p][0]
                uy = nuvs[p][1]

                cx = dtvs[p][0]
                cy = dtvs[p][1]

                v1 = np.array((xmin - px, ymin - py))
                v2 = np.array((xmin - px, ymax - py))
                v3 = np.array((xmax - px, ymin - py))
                v4 = np.array((xmax - px, ymax - py))

                v1, v2, v3, v4 = rot_sort4(v1, v2, v3, v4)
                
                # print(x, y, v1, v2, v3, v4)

                for u in range(nu):
                    sx = cx + (u - (nu/2)) * ux
                    sy = cy + (u - (nu/2)) * uy

                    tx = sx + ux
                    ty = sy + uy

                    if v1[0] * (ty - py) < v1[1] * (tx - px):
                        continue

                    if v4[0] * (sy - py) > v4[1] * (sx - px):
                        break

                    # z1 = line_in_square(px, py, sx, sy, xmin, ymin, xmax, ymax)
                    # z2 = line_in_square(px, py, tx, ty, xmin, ymin, xmax, ymax)

                    # print(x, y, p, u, sx, sy, tx, ty, z1, z2)

                    if not beers_law:
                        val = quad(scipy.LowLevelCallable(line_in_square.ctypes), 0, 1, args = (px, py, sx, sy, ux, uy, xmin, ymin, xmax, ymax))[0]
                    else:
                        val = np.log(quad(scipy.LowLevelCallable(line_in_square_bl.ctypes), 0, 1, args = (px, py, sx, sy, ux, uy, xmin, ymin, xmax, ymax)))[0]

                    if val > 0:
                        points.append(np.array([p*nu + u, x*ny + y, val]))                        

    return np.array(points).T

def get_projection_matrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso, beers_law = False):
    angles = np.linspace(0, 2*np.pi, num=nnp, endpoint=False)

    srcs = np.zeros((nnp, 2), dtype=np.float64)
    dtvs = np.zeros((nnp, 2), dtype=np.float64)
    nuvs = np.zeros((nnp, 2), dtype=np.float64)

    for p, beta in enumerate(angles):
        srcs[p][0] = -lso * np.cos(beta)
        srcs[p][1] = -lso * np.sin(beta)
    
        dtvs[p][0] = (lsd - lso) * np.cos(beta)
        dtvs[p][1] = (lsd - lso) * np.sin(beta)

        nuvs[p][0] = - du * np.sin(beta)
        nuvs[p][1] = du * np.cos(beta)
    
    x = projection_matrix(nx, ny, nnp, nu, dx, dy, srcs, dtvs, nuvs, beers_law)

    return x

if __name__ == "__main__":

    spmat = get_projection_matrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso, False)

    spmat = bsr_array((spmat[2], spmat[:2]), shape=(nnp*nu, nx*ny))
    pickle.dump(spmat, open('./matrixes/A_B0_ref_sparse_test.pkl', 'wb'))

    # np.save(open("ref_100_100_180_855.mat", 'wb'), ref_mat)

    # px = np.sum(ref_mat, axis=(2, 3))
