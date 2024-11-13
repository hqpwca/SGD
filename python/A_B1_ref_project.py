import sys
import ctypes
# import cupy as cp
import numpy as np
import scipy
import math
from scipy.integrate import *
from scipy.sparse import *

import numba as nb
from numba import jit, cfunc, carray
from spline_conv import spline_conv

from config import *

import pickle

c_sig = nb.types.double(nb.types.int32, nb.types.CPointer(nb.types.double))

@cfunc(c_sig)
def line_integral(n, args):
    arg = carray(args, (n,), dtype = np.double)

    #x, px, py, _sx, _sy, ux, uy, xmin, ymin, xmax, ymax = arg

    x, px, py, _sx, _sy, ux, uy, ox, oy, dx, dy = arg

# def line_integral(x, px, py, _sx, _sy, ux, uy, ox, oy, dx, dy):
    eps = 1e-9
    factor = [1.0, 1.0, 2.0, 6.0, 24.0]

    sx = _sx + x*ux
    sy = _sy + x*uy

    theta = np.fabs(np.arctan2(px-sx, py-sy))
    vx = dx * np.sin(theta)
    vy = dy * np.cos(theta)

    l = np.fabs((py-sy)*ox - (px-sx)*oy + px*sy - py*sx)/np.sqrt((px-sx)**2 + (py-sy)**2)

    vecs = np.array([vx, vy, -vx, -vy], dtype = np.float64)
    vecs = vecs[np.fabs(vecs) > eps]

    if l > np.fabs(vx)*2 + np.fabs(vy)*2:
        line_inte = 0.0
    else:
        line_inte = spline_conv(l, 0, vecs) / factor[vecs.shape[0] - 1]

    # print(np.array([px, py, sx, sy, l, line_inte]), vecs)

    return line_inte

# @jit(nopython = True)
# def target_function(x, px, py, sx, sy, ux, uy, xmin, ymin, xmax, ymax):
#     return line_integral(px, py, sx+x*ux, sy+x*uy, xmin, ymin, xmax, ymax)

# @jit(nopython = True)
# def target_function_bl(x, px, py, sx, sy, ux, uy, xmin, ymin, xmax, ymax):
#     return np.power(np.e, line_integral(px, py, sx+x*ux, sy+x*uy, xmin, ymin, xmax, ymax))

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

def projection_matrix(nx, ny, nnp, nu, dx, dy, srcs, dtvs, nuvs):
    points = []
    eps = 1e-10

    for x in range(nx):
        ox = (x - (nx/2)) * dx + 0.5 * dx

        for y in range(ny):
            oy = (y - (ny/2)) * dy + 0.5 * dy

            print(x, y)

            for p in range(nnp):
                px = srcs[p][0]
                py = srcs[p][1]

                ux = nuvs[p][0]
                uy = nuvs[p][1]

                cx = dtvs[p][0]
                cy = dtvs[p][1]

                v1 = np.array((ox - dx - px, oy - dy - py))
                v2 = np.array((ox - dx - px, oy + dy - py))
                v3 = np.array((ox + dx - px, oy - dy - py))
                v4 = np.array((ox + dx - px, oy + dy - py))

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

                    # z1 = line_integral(px, py, sx, sy, xmin, ymin, xmax, ymax)
                    # z2 = line_integral(px, py, tx, ty, xmin, ymin, xmax, ymax)

                    # print(x, y, p, u, sx, sy, tx, ty, z1, z2)

                    val = quad(scipy.LowLevelCallable(line_integral.ctypes), 0, 1, args = (px, py, sx, sy, ux, uy, ox, oy, dx, dy))[0]

                    # val = quad(line_integral, 0, 1, args = (px, py, sx, sy, ux, uy, ox, oy, dx, dy))[0]

                    if val > eps:
                        points.append(np.array([p*nu + u, x*ny + y, val], dtype = np.float64))                        

    return np.array(points).T

def get_projection_matrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso):
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
    
    x = projection_matrix(nx, ny, nnp, nu, dx, dy, srcs, dtvs, nuvs)

    return x

if __name__ == "__main__":

    # nx = 1
    # ny = 1
    # nnp = 2
    # nu = 5
    # lsd = 10
    # lso = 5
    # dx = 1
    # dy = 1
    # du = 1

    if len(sys.argv) < 2:
        print("Usage: python3 [code] [filename]")
        exit(0)

    np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=230)

    spmat = get_projection_matrix(nx, ny, nnp, nu, dx, dy, du, lsd, lso)

    spmat = bsr_array((spmat[2], spmat[:2]), shape=(nnp*nu, nx*ny))

    # xx = (spmat.sum(axis=1).reshape(nnp,nu)*dx*dy)
    # print(xx, file=sys.stderr)

    pickle.dump(spmat, open('./matrixes/A_B1_ref_' + sys.argv[1] + '.pkl', 'wb'))
