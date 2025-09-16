import numpy as np
import scipy
import ctypes
from ctypes import *
# import matplotlib.pyplot as plt
from numba import jit, njit

import pylops
from pylops import LinearOperator

class A_B0_operator_matrix(LinearOperator):
    def __init__(self, nx, ny, nnp, nu, dx, dy, du, lsd, lso, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nx*ny))

        self.vol_shape = (nx, ny)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dx, dy)
        self.blur = (du)

        self.A = scipy.sparse.load_npz("./matrixes/A_B0_sparse_" + str(nx) + '_' + str(ny) + "_" + str(nnp) + "_"+ str(nu) + ".npz")
        self.A = scipy.sparse.csr_matrix(self.A)

    def _matvec(self, x):
        return self.A * x.ravel()

    def _rmatvec(self, y):
        return self.A.T * y.ravel()
    
class A_B1_operator_matrix(LinearOperator):
    def __init__(self, nx, ny, nnp, nu, dx, dy, du, lsd, lso, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nx*ny))

        self.vol_shape = (nx, ny)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dx, dy)
        self.blur = (du)

        self.A = scipy.sparse.load_npz("./matrixes/A_B1_sparse_" + str(nx) + '_' + str(ny) + "_" + str(nnp) + "_"+ str(nu) + ".npz")
        self.A = scipy.sparse.csr_matrix(self.A)

    def _matvec(self, x):
        return self.A * x.ravel()

    def _rmatvec(self, y):
        return self.A.T * y.ravel()
    
class A_H0_operator_matrix(LinearOperator):
    def __init__(self, nl, nc, nnp, nu, dt, du, lsd, lso, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nl*nc))

        self.vol_shape = (nl, nc)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dt)
        self.blur = (du)

        self.A = scipy.sparse.load_npz("./matrixes/A_H0_sparse_" + str(nl) + '_' + str(nc) + "_" + str(nnp) + "_"+ str(nu) + ".npz")
        self.A = scipy.sparse.csr_matrix(self.A)

    def _matvec(self, x):
        return self.A * x.ravel()

    def _rmatvec(self, y):
        return self.A.T * y.ravel()
    
class A_Hl_operator_matrix(LinearOperator):
    def __init__(self, nl, nc, nnp, nu, dt, du, lsd, lso, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nl*nc))

        self.vol_shape = (nl, nc)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dt)
        self.blur = (du)

        self.A = scipy.sparse.load_npz("./matrixes/A_Hl_sparse_" + str(nl) + '_' + str(nc) + "_" + str(nnp) + "_"+ str(nu) + ".npz")
        self.A = scipy.sparse.csr_matrix(self.A)

    def _matvec(self, x):
        return self.A * x.ravel()

    def _rmatvec(self, y):
        return self.A.T * y.ravel()

class A_B0_operator(LinearOperator):
    def __init__(self, nx, ny, nnp, nu, dx, dy, du, lsd, lso, angles, dz = None, drho = None, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nx*ny))

        self.vol_shape = (nx, ny)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dx, dy)
        self.blur = (du)
        self.angles = angles

        angles = angles.astype(np.float64)
        angles_p = angles.ctypes.data_as(POINTER(c_double))

        if dz is not None and drho is None:
            assert 0, "Error: drho should be provided if dz is provided."

        if dz is not None:
            dz = dz.astype(np.float64)
            dz_p = dz.ctypes.data_as(POINTER(c_double))
            drho = drho.astype(np.float64)
            drho_p = drho.ctypes.data_as(POINTER(c_double))

        dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)

        self.cinit = dll.b0_init
        if dz is not None:
            self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double)] # nx, ny, np, nu, dx, dy, du, lsd, lso, angles, dz, drho
        else:
            self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, POINTER(c_double)] # nx, ny, np, nu, dx, dy, du, lsd, lso, angles
        self.cinit.restype = c_void_p

        self.forward = dll.b0_forward_projection
        self.forward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol
        self.forward.restype = c_int

        self.backward = dll.b0_backward_projection
        self.backward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol

        if dz is not None:
            self.cpointer = self.cinit(nx, ny, nnp, nu, dx, dy, du, lsd, lso, angles_p, dz_p, drho_p)
        else:
            self.cpointer = self.cinit(nx, ny, nnp, nu, dx, dy, du, lsd, lso, angles_p)

    def _matvec(self, x):
        y = np.zeros((self.shape[0], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.forward(y_p, x_p, self.cpointer)

        return y

    def _rmatvec(self, y):
        x = np.zeros((self.shape[1], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.backward(y_p, x_p, self.cpointer)

        return x
    
class A_B1_operator(LinearOperator):
    def __init__(self, nx, ny, nnp, nu, dx, dy, du, lsd, lso, angles, dz = None, drho = None, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nx*ny))

        self.vol_shape = (nx, ny)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dx, dy)
        self.blur = (du)
        self.angles = angles

        angles = angles.astype(np.float64)
        angles_p = angles.ctypes.data_as(POINTER(c_double))

        if dz is not None and drho is None:
            assert 0, "Error: drho should be provided if dz is provided."

        if dz is not None:
            dz = dz.astype(np.float64)
            dz_p = dz.ctypes.data_as(POINTER(c_double))
            drho = drho.astype(np.float64)
            drho_p = drho.ctypes.data_as(POINTER(c_double))

        dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)

        self.cinit = dll.b1_init
        if dz is not None:
            self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double)] # nx, ny, np, nu, dx, dy, du, lsd, lso, angles, dz, drho
        else:
            self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, POINTER(c_double)] # nx, ny, np, nu, dx, dy, du, lsd, lso, angles
        self.cinit.restype = c_void_p

        self.forward = dll.b1_forward_projection
        self.forward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol
        self.forward.restype = c_int

        self.backward = dll.b1_backward_projection
        self.backward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol

        if dz is not None:
            self.cpointer = self.cinit(nx, ny, nnp, nu, dx, dy, du, lsd, lso, angles_p, dz_p, drho_p)
        else:
            self.cpointer = self.cinit(nx, ny, nnp, nu, dx, dy, du, lsd, lso, angles_p)

    def _matvec(self, x):
        y = np.zeros((self.shape[0], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.forward(y_p, x_p, self.cpointer)

        return y

    def _rmatvec(self, y):
        x = np.zeros((self.shape[1], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.backward(y_p, x_p, self.cpointer)

        return x

class A_H0_operator(LinearOperator):
    def __init__(self, nx, ny, nnp, nu, dt, du, lsd, lso, angles, dz, drho, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nx*ny))

        self.vol_shape = (nx, ny)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dt)
        self.blur = (du)
        self.angles = angles

        angles = angles.astype(np.float64)
        angles_p = angles.ctypes.data_as(POINTER(c_double))

        if dz is not None and drho is None:
            assert 0, "Error: drho should be provided if dz is provided."
        
        if dz is not None:
            dz = dz.astype(np.float64)
            dz_p = dz.ctypes.data_as(POINTER(c_double))
            drho = drho.astype(np.float64)
            drho_p = drho.ctypes.data_as(POINTER(c_double))

        dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)

        self.cinit = dll.h0_init
        if dz is not None:
            self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double)] # nx, ny, np, nu, dt, du, lsd, lso, angles, dz, drho
        else:
            self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, POINTER(c_double)] # nx, ny, np, nu, dt, du, lsd, lso, angles
        self.cinit.restype = c_void_p

        self.forward = dll.h0_forward_projection
        self.forward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol
        self.forward.restype = c_int

        self.backward = dll.h0_backward_projection
        self.backward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol

        if dz is not None:
            self.cpointer = self.cinit(nx, ny, nnp, nu, dt, du, lsd, lso, angles_p, dz_p, drho_p)
        else:
            self.cpointer = self.cinit(nx, ny, nnp, nu, dt, du, lsd, lso, angles_p)

    def _matvec(self, x):
        y = np.zeros((self.shape[0], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.forward(y_p, x_p, self.cpointer)

        return y

    def _rmatvec(self, y):
        x = np.zeros((self.shape[1], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.backward(y_p, x_p, self.cpointer)

        return x

class A_Hl_operator(LinearOperator):
    def __init__(self, nx, ny, nnp, nu, dt, du, lsd, lso, angles, dz, drho, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nx*ny))

        self.vol_shape = (nx, ny)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dt)
        self.blur = (du)
        self.angles = angles

        angles = angles.astype(np.float64)
        angles_p = angles.ctypes.data_as(POINTER(c_double))

        if dz is not None and drho is None:
            assert 0, "Error: drho should be provided if dz is provided."
        
        if dz is not None:
            dz = dz.astype(np.float64)
            dz_p = dz.ctypes.data_as(POINTER(c_double))
            drho = drho.astype(np.float64)
            drho_p = drho.ctypes.data_as(POINTER(c_double))

        dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)

        self.cinit = dll.hl_init
        if dz is not None:
            self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double)] # nx, ny, np, nu, dt, du, lsd, lso, angles, dz, drho
        else:
            self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, POINTER(c_double)] # nx, ny, np, nu, dt, du, lsd, lso, angles
        self.cinit.restype = c_void_p

        self.forward = dll.hl_forward_projection
        self.forward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol
        self.forward.restype = c_int

        self.backward = dll.hl_backward_projection
        self.backward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol

        if dz is not None:
            self.cpointer = self.cinit(nx, ny, nnp, nu, dt, du, lsd, lso, angles_p, dz_p, drho_p)
        else:
            self.cpointer = self.cinit(nx, ny, nnp, nu, dt, du, lsd, lso, angles_p)

    def _matvec(self, x):
        y = np.zeros((self.shape[0], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.forward(y_p, x_p, self.cpointer)

        return y

    def _rmatvec(self, y):
        x = np.zeros((self.shape[1], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.backward(y_p, x_p, self.cpointer)

        return x

class A_SF_operator(LinearOperator):
    def __init__(self, nx, ny, nnp, nu, dx, dy, du, lsd, lso, angles, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu, nx*ny))

        self.vol_shape = (nx, ny)
        self.proj_shape = (nnp, nu)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dx, dy)
        self.blur = (du)
        self.angles = angles

        angles = angles.astype(np.float64)
        angles_p = angles.ctypes.data_as(POINTER(c_double))

        dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)

        self.cinit = dll.SF_init
        self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, POINTER(c_double)] # nx, ny, np, nu, dx, dy, du, lsd, lso
        self.cinit.restype = c_void_p

        self.forward = dll.SF_forward_projection
        self.forward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol
        self.forward.restype = c_int

        self.backward = dll.SF_backward_projection
        self.backward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p] # double *proj, double *vol

        self.cpointer = self.cinit(nx, ny, nnp, nu, dx, dy, du, lsd, lso, angles_p)

    def _matvec(self, x):
        y = np.zeros((self.shape[0], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.forward(y_p, x_p, self.cpointer)

        return y

    def _rmatvec(self, y):
        x = np.zeros((self.shape[1], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.backward(y_p, x_p, self.cpointer)

        return x

class A_B1_3D_operator(LinearOperator):
    def __init__(self, nx, ny, nz, nnp, nu, nv, dx, dy, dz, du, dv, lsd, lso, angles, dtype=np.float64):
        super().__init__(dtype=dtype, shape=(nnp*nu*nv, nx*ny*nz))

        self.vol_shape = (nz, ny, nx)
        self.proj_shape = (nnp, nu, nv)
        self.lsd = lsd
        self.lso = lso
        self.dimension = (dx, dy, dz)
        self.blur = (du, dv)
        self.angles = angles

        angles = angles.astype(np.float64)
        angles_p = angles.ctypes.data_as(POINTER(c_double))

        dll = CDLL('../build/libnetwork.so', mode = RTLD_GLOBAL)

        self.cinit = dll.b1_3d_init
        self.cinit.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, c_double, c_double, POINTER(c_double)]
        self.cinit.restype = c_void_p

        self.forward = dll.b1_3d_forward_projection
        self.forward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p]
        self.forward.restype = c_int

        self.backward = dll.b1_3d_backward_projection
        self.backward.argtypes = [POINTER(c_double), POINTER(c_double), c_void_p]

        self.cpointer = self.cinit(nx, ny, nz, nnp, nu, nv, dx, dy, dz, du, dv, lsd, lso, angles_p)

    def _matvec(self, x):
        y = np.zeros((self.shape[0], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.forward(y_p, x_p, self.cpointer)

        return y

    def _rmatvec(self, y):
        x = np.zeros((self.shape[1], ), dtype=self.dtype)

        x_p = x.ctypes.data_as(POINTER(c_double))
        y_p = y.ctypes.data_as(POINTER(c_double))

        self.backward(y_p, x_p, self.cpointer)

        return x
