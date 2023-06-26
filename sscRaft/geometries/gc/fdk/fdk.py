from ....rafttypes import *
import numpy as np
import ctypes
import ctypes.util


def set_experiment( x, y, z, dx, dy, dz, nx, ny, nz, 
                    h, v, dh, dv, nh, nv,
                    D, Dsd, beta_max, dbeta, nbeta,
                    fourier):

    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta,
                fourier = fourier)

    return lab


def fdk(tomogram: np.ndarray, dic: dict = {}) -> np.ndarray:
    """Computes the Reconstruction of a Conical Sinogram using the Filtered Backprojection method for conical rays(FDK).
    GPU function.

    Args:
        data (ndarray): Cone beam projection tomogram. The axes are [angle, slices, lenght].
        dic (dictionary): Dictionary with the experiment info.

    Returns:
        (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [x, y, z].

    Dictionary parameters:
        *``dic['gpu']`` (ndarray): List of gpus for processing. Defaults to [0].
        *``dic['z1[m]']`` (float): Source-sample distance in meters. Defaults to 500e-3.
        *``dic['z1+z2[m]']`` (float): Source-detector distance in meters. Defaults to 1.0.
        *``dic['detectorPixel[m]']`` (float): Detector pixel size in meters. Defaults to 1.44e-6.
        *``dic['reconSize']`` (int): Reconstruction dimension. Defaults to data shape[0].
        *``dic['fourier']`` (bool): Type of filter for reconstruction. True = Fourier, False = Convolution.
    """

    # recon = data 
    D, Dsd = dic['z1[m]'], dic['z1+z2[m]']

    dh, dv = dic['detectorPixel[m]'], dic['detectorPixel[m]']
    nh, nv = int(tomogram.shape[2]), int(tomogram.shape[0])
    h, v   = nh*dh/2, nv*dv/2
  
    beta_max, nbeta = 2*np.pi, int(tomogram.shape[1])
    dbeta           = beta_max/nbeta

    magn       = D/Dsd
    dx, dy, dz = dh*magn, dh*magn, dv*magn
    nx, ny, nz = int(dic['reconSize']), int(dic['reconSize']), int(dic['reconSize'])
    x, y, z    = dx*nx/2, dy*ny/2, dz*nz/2

    fourier = dic['fourier']

    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta,
                fourier = fourier )

    time = np.zeros(2)
    time = np.ascontiguousarray(time.astype(np.float64))
    time_p = time.ctypes.data_as(ctypes.c_void_p)

    gpus = np.array(dic['gpu'])
    ndev = len(gpus)
    gpus = np.ascontiguousarray(gpus.astype(np.intc))
    gpus_p = gpus.ctypes.data_as(ctypes.c_void_p)

    proj = np.ascontiguousarray(tomogram.astype(np.float32))
    proj_p = proj.ctypes.data_as(ctypes.c_void_p)

    recon = np.zeros((lab.nz, lab.ny, lab.nx))
    recon = np.ascontiguousarray(recon.astype(np.float32))
    recon_p = recon.ctypes.data_as(ctypes.c_void_p)

    libraft.gpu_fdk(lab, recon_p, proj_p, gpus_p, int(ndev), time_p)

    print("Time of execution: \n \n"+"Phantom size"+str(lab.nx)+"\n Nbeta"+str(lab.nbeta)+"\n ngpus"+str(ndev)+"\n"+str(time))

    return recon
