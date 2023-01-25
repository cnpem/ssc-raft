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
                    D, Dsd, beta_max, dbeta, nbeta,
                    rings ):

    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta,
                rings = rings)

    return lab


def fdk(lab, proj, gpus):

    print("Reconstruction...")

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



def reconstruction_fdk( experiment, data, flat = {}, dark = {}):

    """Computes the Reconstruction of a Conical Sinogram using the Filtered Backprojection method for conical rays(FDK).

    Args:
        data (ndarray): Cone beam projection tomogram. The axes are [angle, slices, lenght].

        flat (ndarray): Single cone beam ray projection. The axes are [slices, lenght].

        dark (ndarray): Single dark projection. The axes are [slices, lenght].

        experiment (dictionary): Dictionary with the experiment info.
            z1 (float): Source-sample distance in meters.
            z2 (float): Source-detector distance in meters.
            pixel (float): Detector pixel size in meters.
            n (int): Reconstruction dimension.
            gpus (ndarray): List of gpus for processing.
            apply_rings (bool): Flag for application of rings removal algorithm.
            normalize (bool): Flag for normalization of projection data.
            padding (int): Number of elements for horizontal zero-padding.

    Returns:
        (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [x, y, z].
    """

    D, Dsd = experiment['z1'], experiment['z2']

    dh, dv = experiment['pixel'], experiment['pixel']
    nh, nv = int(data.shape[2] + experiment['padding']), int(data.shape[1])
    h, v = nh*dh/2, nv*dv/2
  
    beta_max, nbeta = 2*np.pi, int(data.shape[0])
    dbeta = beta_max/nbeta

    magn = D/Dsd
    dx, dy, dz = dh*magn, dh*magn, dv*magn
    nx, ny, nz = int(experiment['n']), int(experiment['n']), int(experiment['n'])
    x, y, z = dx*nx/2, dy*ny/2, dz*nz/2

    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta,
                rings = int(experiment['apply_rings']) )

    if(experiment['normalize']):
        data = normalize_fdk(data, flat, dark, experiment['padding'])
    else:
        data = np.swapaxes(0,1,data)
    
    recon = fdk(lab, data, experiment['gpus'])

    print('fdk')

    return recon


def normalize_fdk(tomo, flat, dark, padding):
    print('Normalize ....')

    tomo[:,:11,:] = 1.0
    flat[:11,:] = 1.0
    dark[:11,:] = 0.0

    flat = flat - dark
    for i in range(tomo.shape[0]):
        tomo[i,:,:] = tomo[i,:,:] - dark
    
    median_projection = np.median(tomo[tomo > 0])
    median_flat_field = np.median(flat[flat > 0])

    tomo[tomo ==0] = 1.0
    tomo[tomo == np.max(tomo)] = median_projection
    flat[flat == 0] = 1.0
    flat[flat == np.max(flat)] = median_flat_field

    for i in range(tomo.shape[0]):
        tomo[i,:,:] = -np.log(tomo[i,:,:]/flat)

    tomo[np.isinf(tomo)] = 0.0
    tomo[np.isnan(tomo)] = 0.0

    shift = find_rotation_axis_fdk(tomo)
    if padding < 2*np.abs(shift):
        padding = 2*shift
    padd = padding -2*np.abs(shift)

    proj = np.zeros((tomo.shape[1], tomo.shape[0], tomo.shape[2]+padding))

    if(shift < 0):
        proj[:,:,padd//2 + 2*np.abs(shift):tomo.shape[2]+padd//2 + 2*np.abs(shift)] = np.swapaxes(tomo,0,1)
    else:
        proj[:,:,padd//2:tomo.shape[2]+padd//2] = np.swapaxes(tomo,0,1)

    print('Projections ok!')
    print('Projection shape =', proj.shape[0], proj.shape[1], proj.shape[2])
    return proj

def find_rotation_axis_fdk(tomo, nx_search=1000, nx_window=5, nsinos=None):
    """Searches for the rotation axis index in axis 2 (x variable).
    It minimizes the symmetry error.
    Works with parallel, fan and cone beam sinograms (proof: to do).
    It is assumed the center of rotation is between two pixels. 
    It might be interesting to implement an alternative where the center is ON a pixel (better suitted for odd number of angles/projections).

    Parameters
    ----------
    tomo : 3 dimensional array_like object
        Raw tomogram.
        tomo[m, i, j] selects the value of the pixel at the i-row and j-column in the projection m.
    nx_search : int, optional
        Width of the search. 
        If the center of rotation is not in the interval [nx_search-nx//2; nx_search+nx//2] this function will return a wrong result.
        Default is nx_search=400.
nx_window : int, optional
        How much of the sinogram will be used in the axis 2.
        Default is nx_window=400.
    nsinos : int or None, optional
        Number of sinograms to avarege over.
        Default is None, which results in nsinos = nz//2, where nz = tomo.shape[1].

    Returns
    -------
    deviation : int
        Number of pixels representing the deviation of the center of rotation from the middle of the tomogram (about the axis=2).
        The integer "deviation + nx//2" is the pixel index which minimizes the symmetry error in the tomogram.
    
    Raises
    ------
    ValueError
        If the number of angles/projections is not an even number.
    """
    ntheta, nz, nx = tomo.shape

    diff_symmetry = np.ones(2*nx_search) # começando em zero para podermos fazer um gráfico de semilogy.

    if nsinos is None:
        nsinos = nz//20
    if nsinos == 0:
        nsinos = nz
    
    for k in range(nsinos):
        for i in range(2*nx_search):
            center_idx = (nx//2)-nx_search+i

            if (ntheta%2 == 0):
                sino_esq = tomo[:ntheta//2, k+(nz//2), center_idx-nx_window : center_idx]
                sino_dir = np.flip(tomo[ntheta//2:, k+(nz//2), center_idx : center_idx+nx_window], axis=1)
            else:
                sino_esq = tomo[:ntheta//2, k+(nz//2), center_idx-nx_window : center_idx]
                sino_dir = np.flip(tomo[ntheta//2:ntheta-1, k+(nz//2), center_idx : center_idx+nx_window], axis=1)
            

            mean_sinos = np.linalg.norm(sino_esq + sino_dir)/2
            diff_symmetry[i] += np.linalg.norm(sino_esq - sino_dir) / mean_sinos
    
    deviation = np.argmin(diff_symmetry) - nx_search

    print('Shift :', deviation)

    return deviation
