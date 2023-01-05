from random import betavariate
from ....rafttypes import *
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import imageio
import ctypes
import ctypes.util



def set_experiment( x, y, z, dx, dy, dz, nx, ny, nz, 
                    h, v, dh, dv, nh, nv,
                    D, Dsd, beta_max, dbeta, nbeta):
    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta)

    return lab


def reconstruction_fdk_(     lab, proj, gpus):

    print("Reconstruction...")

    recon = fdk(lab, proj, gpus)

    return recon


def fdk(lab, proj, gpus):

    time = np.zeros(2)
    time = np.ascontiguousarray(time.astype(np.float64))
    time_p = time.ctypes.data_as(ctypes.c_void_p)

    ndev = len(gpus)
    gpus = np.ascontiguousarray(gpus.astype(np.int32))
    gpus_p = gpus.ctypes.data_as(ctypes.c_void_p)

    proj = np.ascontiguousarray(proj.astype(np.float32))
    proj_p = proj.ctypes.data_as(ctypes.c_void_p)

    recon = np.zeros((lab.nz, lab.ny, lab.nx))
    recon = np.ascontiguousarray(recon.astype(np.float32))
    recon_p = recon.ctypes.data_as(ctypes.c_void_p)

    libraft.gpu_fdk(lab, recon_p, proj_p, gpus_p, int(ndev), time_p)

    print("Time of execution: \n \n"+"Phantom size"+str(lab.nx)+"\n Nbeta"+str(lab.nbeta)+"\n ngpus"+str(ndev)+"\n              "+str(time))

    return recon


def reconstruction_fdk( data, experiment):

    """Computes the Reconstruction of a Conical Sinogram using the Filtered Backprojection method for conical rays(FDK).

    Args:
        data (ndarray): Cone beam tomogram. The axes are [slices, angle, lenght].
        experiment (dictionary): Dictionary with the experiment info.
            z1 (float): source-sample distance in meters
            z2 (float): source-detector distance in meters
            pix (float): detector pixel size in meters
            n (int): reconstruction dimension
            gpus (ndarray): List of gpus for processing

    Returns:
        (ndarray): Reconstructed object with dimension n^3 (3D). The axes are [x, y, z].
    """

    D, Dsd = experiment['z1'], experiment['z2']

    dh, dv = experiment['pix'], experiment['pix']
    nh, nv = data.shape[2], data.shape[0]
    h, v = nh*dh/2, nv*dv/2
  
    beta_max, nbeta = 2*np.pi, data.shape[1]
    dbeta = beta_max/nbeta

    magn = D/Dsd
    dx, dy, dz = dh*magn, dh*magn, dv*magn
    nx, ny, nz = experiment['n'], experiment['n'], experiment['n']
    x, y, z = dx*nx/2, dy*ny/2, dz*nz/2

    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta)
    
    recon = reconstruction_fdk_(lab, data, experiment['gpus'])

    print('fdk')

    return recon


