# import cupy as cp
import sys
import numpy as np

import pickle

from scipy import LowLevelCallable
from scipy.io import loadmat
from scipy.integrate import quad

import numba as nb
from numba import jit, njit
from numba import cfunc, carray

np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=10000, suppress=True)

# https://pmc.ncbi.nlm.nih.gov/articles/PMC3426508/#APP1

def _analytical_forbild_phantom(resolution, ear):
    """Analytical description of FORBILD phantom.

    Parameters
    ----------
    resolution : bool
        If ``True``, insert a small resolution test pattern to the left.
    ear : bool
        If ``True``, insert an ear-like structure to the right.
    """
    sha = 0.2 * np.sqrt(3)
    y016b = -14.294530834372887
    a16b = 0.443194085308632
    b16b = 3.892760834372886

    E = [[-4.7, 4.3, 1.79989, 1.79989, 0, 0.010, 0],  # 1
         [4.7, 4.3, 1.79989, 1.79989, 0, 0.010, 0],  # 2
         [-1.08, -9, 0.4, 0.4, 0, 0.0025, 0],  # 3
         [1.08, -9, 0.4, 0.4, 0, -0.0025, 0],  # 4
         [0, 0, 9.6, 12, 0, 1.800, 0],  # 5
         [0, 8.4, 1.8, 3.0, 0, -1.050, 0],  # 7
         [1.9, 5.4, 0.41633, 1.17425, -31.07698, 0.750, 0],  # 8
         [-1.9, 5.4, 0.41633, 1.17425, 31.07698, 0.750, 0],  # 9
         [-4.3, 6.8, 1.8, 0.24, -30, 0.750, 0],  # 10
         [4.3, 6.8, 1.8, 0.24, 30, 0.750, 0],  # 11
         [0, -3.6, 1.8, 3.6, 0, -0.005, 0],  # 12
         [6.39395, -6.39395, 1.2, 0.42, 58.1, 0.005, 0],  # 13
         [0, 3.6, 2, 2, 0, 0.750, 4],  # 14
         [0, 9.6, 1.8, 3.0, 0, 1.800, 4],  # 15
         [0, 0, 9.0, 11.4, 0, 0.750, 3],  # 16a
         [0, y016b, a16b, b16b, 0, 0.750, 1],  # 16b
         [0, 0, 9.0, 11.4, 0, -0.750, ear],  # 6
         [9.1, 0, 4.2, 1.8, 0, 0.750, 1], # R_ear
         [-3, 0, 3, 3, 0, 0.8, 4],
         [-4, 0, 3, 3, 0, 0.8, 4],
         [-5, 0, 3, 3, 0, 0.8, 4],
         [-4, -6, 3, 3, 0, 0.8, 4],
         [-4, -6, 3, 3, 0, 0.8, 4],
         [-4, -6, 3, 3, 0, 0.8, 4],
         [-4, -6, 3, 3, 0, 0.8, 4],
         [-4, -6, 3, 3, 0, 0.8, 4],
         [-4, -6, 3, 3, 0, 0.8, 4]]  
    E = np.array(E)

    # generate the air cavities in the right ear
    cavity1 = np.arange(8.8, 5.6, -0.4)[:, None]
    cavity2 = np.zeros([9, 1])
    cavity3_7 = np.ones([53, 1]) * [0.15, 0.15, 0, -1.800, 0]

    for j in range(1, 4):
        kj = 8 - 2 * int(np.floor(j / 3))
        dj = 0.2 * int(np.mod(j, 2))

        cavity1 = np.vstack((cavity1,
                             cavity1[0:kj] - dj,
                             cavity1[0:kj] - dj))
        cavity2 = np.vstack((cavity2,
                             j * sha * np.ones([kj, 1]),
                             -j * sha * np.ones([kj, 1])))

    E_cavity = np.hstack((cavity1, cavity2, cavity3_7))

    # generate the left ear (resolution pattern)
    r = 3.0

    x0 = -7.0
    y0 = -1.0
    d0_xy = 0.04*r

    d_xy = [0.0357*r, 0.0312*r, 0.0278*r, 0.0250*r]
    ab = 0.5 * np.ones([5, 1]) * d_xy
    ab = ab.T.ravel()[:, None] * np.ones([1, 4])
    abr = ab.T.ravel()[:, None]

    leftear4_7 = np.hstack([abr, abr, np.ones([80, 1]) * [0, 0.75, 0]])

    x00 = np.zeros([0, 1])
    y00 = np.zeros([0, 1])
    for i in range(1, 5):
        y00 = np.vstack((y00,
                         (y0 + np.arange(0, 5) * 2 * d_xy[i - 1])[:, None]))
        x00 = np.vstack((x00,
                         (x0 + 2 * (i - 1) * d0_xy) * np.ones([5, 1])))

    x00 = x00 * np.ones([1, 4])
    x00 = x00.T.ravel()[:, None]
    y00 = np.vstack([y00, y00 + 12 * d0_xy,
                     y00 + 24 * d0_xy, y00 + 36 * d0_xy])

    leftear = np.hstack([x00, y00, leftear4_7])

    C = [[1.2, 1.2, 0.27884, 0.27884, 0.60687, 0.60687, 0.2,
          0.2, -2.605, -2.605, -10.71177, y016b + 10.71177, 8.88740, -0.21260, 
          0.1, 0.1, 2, 2, 
          0.1, 0.1, 2, 2, 
          0.1, 0.1, 2, 2,
          0.1, 0.1, 2, 2, 
          0.1/2*np.sqrt(3), 0.1/2*np.sqrt(3), 2, 2, 
          0.1/2, 0.1/2, 2, 2,
          0.1, 0.1, 2, 2, 
          0.1/2*np.sqrt(3), 0.1/2*np.sqrt(3), 2, 2, 
          0.1/2, 0.1/2, 2, 2],
         [0, 180, 90, 270, 90, 270, 0,
          180, 15, 165, 90, 270, 0, 0, 
          0, 180, 90, 270, 
          0, 180, 90, 270, 
          0, 180, 90, 270,
          0, 180, 90, 270,
          0+30, 180+30, 90+30, 270+30,
          0+60, 180+60, 90+60, 270+60,
          0+90, 180+90, 90+90, 270+90,
          0+120, 180+120, 90+120, 270+120,
          0+150, 180+150, 90+150, 270+150,]]
    C = np.array(C)

    if not resolution and not ear:
        phantomE = E[:17, :]
        phantomC = C[:, :12]
    elif not resolution and ear:
        phantomE = np.vstack([E, E_cavity])
        phantomC = C
    elif resolution and not ear:
        phantomE = np.vstack([leftear, E[:17, :]])
        phantomC = C[:, :12]
    else:
        phantomE = np.vstack([leftear, E, E_cavity])
        phantomC = C

    new_phantomE = []
    for p in phantomE:
        x0 = p[0] * 4
        y0 = p[1] * 4
        a = p[2] * 4
        b = p[3] * 4
        phi = p[4]*np.pi/180
        f = p[5]
        nclip = p[6]

        DQ = np.array([np.cos(phi) / a, np.sin(phi) / a, -np.sin(phi) / b, np.cos(phi) / b])

        p1 = np.array([x0, y0, a, b, p[4], f, nclip])
        pl = np.append(p1, DQ)

        new_phantomE.append(pl)

    new_phantomE = np.array(new_phantomE)

    newC = []

    newC.append(C[0]*4)
    new_angle = np.array(C[1]) * np.pi / 180
    newC.append(C[1])
    newC.append(np.cos(new_angle))
    newC.append(np.sin(new_angle))

    newC = np.array(newC)

    phantomE = new_phantomE
    phantomC = newC

    return phantomE, phantomC

phantomE, phantomC = _analytical_forbild_phantom(False, True)

def discrete_phantom(xcoord, ycoord):
    image = np.zeros(xcoord.shape)
    nclipinfo = 0

    # Loop over each row in phantomE
    for k in range(phantomE.shape[0]):
        Vx0 = np.array([xcoord.ravel() - phantomE[k, 0], ycoord.ravel() - phantomE[k, 1]])
        D = np.array([[1 / phantomE[k, 2], 0], [0, 1 / phantomE[k, 3]]])
        phi = phantomE[k, 4] * np.pi / 180
        Q = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        f = phantomE[k, 5]
        nclip = int(phantomE[k, 6])

        # Calculate the ellipse equation and find indices inside the ellipse
        equation1 = np.sum((D @ Q @ Vx0)**2, axis=0)
        i = np.where(equation1 <= 1.0)[0]

        # Handle clipping if nclip > 0
        if nclip > 0:
            for j in range(nclip):
                nclipinfo += 1
                d = phantomC[0, nclipinfo - 1]
                equation2 = np.dot([phantomC[2, nclipinfo - 1], phantomC[3, nclipinfo - 1]], Vx0)
                i = i[np.where(equation2[i] < d)]

        image.ravel()[i] += f

    return image

# @njit
def batch_line_integral(thetas, scoord):
    sinth = np.sin(thetas)
    costh = np.cos(thetas)

    eps = 1e-10
    nc = 0
    mask = np.zeros(thetas.shape)

    for k in range(phantomC.shape[1]):
        tmp = np.fabs(-sinth*phantomC[2, k]+costh*phantomC[3, k])
        mask[tmp < eps] = eps

    thetas = thetas + mask
    sino =  np.zeros(scoord.shape)
    sinth = np.sin(thetas)
    costh = np.cos(thetas)

    sx = scoord * costh
    sy = scoord * sinth

    for p in phantomE:
        x0 = p[0]
        y0 = p[1]
        a = p[2]
        b = p[3]
        phi = p[4]*np.pi/180
        f = p[5]
        nclip = p[6]

        s0 = np.array([sx-x0, sy-y0])

        DQ = np.array([[p[7], p[8]], [p[9], p[10]]])

        DQthp = DQ @ np.array([-sinth, costh])
        DQs0 = DQ @ s0

        A = np.sum(DQthp**2, axis=0)
        B = 2 * np.sum(DQthp * DQs0, axis=0)
        C = np.sum(DQs0**2, axis=0) - 1

        equation = B**2 - 4 * A * C
        i = np.nonzero(equation > 0)

        tp = 0.5 * (-B[i] + np.sqrt(equation[i])) / A[i]
        tq = 0.5 * (-B[i] - np.sqrt(equation[i])) / A[i]

        for j in range(int(nclip)):
            d = phantomC[0, nc]
            xp = sx[i] - tp * sinth[i]
            yp = sy[i] + tp * costh[i]
            xq = sx[i] - tq * sinth[i]
            yq = sy[i] + tq * costh[i]
            tz = d - phantomC[2, nc] * s0[0, i] - phantomC[3, nc] * s0[1, i]
            tz = tz / (-sinth[i] * phantomC[2, nc] + costh[i] * phantomC[3, nc])
            tz = tz.ravel()
            equation2 = (xp - x0) * phantomC[2, nc] + (yp - y0) * phantomC[3, nc]
            equation3 = (xq - x0) * phantomC[2, nc] + (yq - y0) * phantomC[3, nc]
            m1 = np.nonzero(equation2 >= d)
            tp[m1] = tz[m1]
            m2 = np.nonzero(equation3 >= d)
            tq[m2] = tz[m2]
            nc += 1

        sinok = f * np.abs(tp - tq)
        sino[i] += sinok

    return sino.reshape(thetas.shape)

@njit
def forbild_line_integral(theta, scoord):
    sinth = np.sin(theta)
    costh = np.cos(theta)

    eps = 1e-10
    nc = 0
    mask = 0

    for k in range(phantomC.shape[1]):
        tmp = np.fabs(-sinth*phantomC[2, k]+costh*phantomC[3, k])
        if tmp < eps: mask = eps

    theta += mask

    sino = 0.0
    sinth = np.sin(theta)
    costh = np.cos(theta)

    sx = scoord * costh
    sy = scoord * sinth

    for k in range(phantomE.shape[0]):
        p = phantomE[k]

        x0 = p[0]
        y0 = p[1]
        a = p[2]
        b = p[3]
        phi = p[4]*np.pi/180
        f = p[5]
        nclip = p[6]

        s0 = np.array([sx-x0, sy-y0])

        DQ = np.array([[p[7], p[8]], [p[9], p[10]]])

        DQthp = DQ @ np.array([-sinth, costh])
        DQs0 = DQ @ s0

        A = np.sum(DQthp**2)
        B = 2 * np.sum(DQthp * DQs0)
        C = np.sum(DQs0**2) - 1

        equation = B**2 - 4 * A * C
        if equation < 0:
            nc += int(nclip)
            continue

        tp = 0.5 * (-B + np.sqrt(equation)) / A
        tq = 0.5 * (-B - np.sqrt(equation)) / A

        for j in range(int(nclip)):
            d = phantomC[0, nc]
            xp = sx - tp * sinth
            yp = sy + tp * costh
            xq = sx - tq * sinth
            yq = sy + tq * costh
            tz = d - phantomC[2, nc] * s0[0] - phantomC[3, nc] * s0[1]
            tz = tz / (-sinth * phantomC[2, nc] + costh * phantomC[3, nc])
            # print(np.array([d, xp, yp, xq, yq, tz]))
            if (xp - x0) * phantomC[2, nc] + (yp - y0) * phantomC[3, nc] >= d:
                tp = tz
            if (xq - x0) * phantomC[2, nc] + (yq - y0) * phantomC[3, nc] >= d:
                tq = tz
            nc += 1

        sinok = f * np.fabs(tp - tq)

        # print(k, np.array([tp, tq, nc]))
        sino += sinok

    return sino

def calc_poisson_noise(x, N0):
    noise = np.random.poisson(lam = N0 * np.exp(-x/100.0))
    return noise

def forbild_sinogram_noise(nnp, nu, du, lsd, lso):
    sino = np.zeros((nnp, nu), dtype=np.float64)

    angles = np.linspace(0, 2*np.pi, num=nnp, endpoint=False)
    for p in range(nnp):
        for iu in range(nu):
            su = - (nu*du) / 2 + iu*du

            samples, step = np.linspace(su, su+du, 200, endpoint=False, retstep=True)
            samples += step / 2

            theta = angles[p] + np.pi/2 + np.arctan2(samples, lsd)
            scoord = samples * lso / np.sqrt(lsd**2 + samples**2)

            # samples = np.vectorize(forbild_line_integral)(theta, scoord)
            samples = batch_line_integral(theta, scoord)
            # if not np.allclose(samples, samples2):
            #     print(p, iu)
            #     print(samples)
            #     print(samples2)
            #     assert False

            samples = calc_poisson_noise(samples, 100.0)

            sino[p][iu] = 100.0 * np.log(20000.0 / np.sum(samples))
        print(p, flush=True)

    return sino


c_sig = nb.types.double(nb.types.int32, nb.types.CPointer(nb.types.double))

@cfunc(c_sig)
def forbild_line_quad(n, args):
    arg = carray(args, (n,), dtype = np.double)
    x, su, du, lso, lsd, la = arg

    u = su + x * du

    theta = la + np.pi/2 + np.arctan2(u, lsd)
    scoord = u * lso / np.sqrt(lsd**2 + u**2)

    return forbild_line_integral(theta, scoord)

@cfunc(c_sig)
def forbild_line_quad_beerslaw(n, args):
    arg = carray(args, (n,), dtype = np.double)
    x, su, du, lso, lsd, la = arg

    u = su + x * du

    theta = la + np.pi/2 + np.arctan2(u, lsd)
    scoord = u * lso / np.sqrt(lsd**2 + u**2)

    return np.exp(forbild_line_integral(theta, scoord))


def forbild_sinogram(nnp, nu, du, lsd, lso, beers_law = False):
    sino = np.zeros((nnp, nu), dtype=np.float64)

    angles = np.linspace(0, 2*np.pi, num=nnp, endpoint=False)
    for p in range(nnp):
        for iu in range(nu):
            su = - (nu*du) / 2 + iu*du
            if beers_law:
                sino[p][iu] = np.log(quad(LowLevelCallable(forbild_line_quad_beerslaw.ctypes), 0, 1, args=(su, du, lso, lsd, angles[p]))[0])
            else:
                sino[p][iu] = quad(LowLevelCallable(forbild_line_quad.ctypes), 0, 1, args=(su, du, lso, lsd, angles[p]))[0]
        print(p, flush=True)

    return sino

if __name__ == "__main__":
    nx = 100
    ny = 100
    nnp = 180
    nu = 855
    lsd = 156.25
    lso = 78.125
    dx = 1
    dy = 1
    du = 0.78125

    # print(phantomC)
    # print(forbild_line_integral(0, 0))
    
    sino = forbild_sinogram(nnp, nu, du, lsd, lso, True)

    np.save(open("FORBILD_sinogram.dat", 'wb'), sino)

