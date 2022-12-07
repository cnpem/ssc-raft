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
                    fourier):
    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta,
                fourier = fourier)
    return lab


def reconstruction(     lab, proj, gpus, 
                        method = 'fdk', save_path = '', 
                        save = False, vis = False):

    print("Reconstruction...")

    # proj = np.load(path_tomo)
    print("Projection shape  = "+ str(proj.shape))

    recon = fdk(lab, proj, gpus)

    print("Reconstruction Shape  = "+ str(recon.shape))

    if save:
        print('Saving Results...')
        np.save(save_path+"/recon_"+str(lab.nx)+"_"+ str(lab.nbeta)+".npy", recon)    

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

    return recon#,proj


def reconstruction_fdk( data, experiment):

    D, Dsd = experiment['z1'], experiment['z2']

    dh, dv = experiment['detector']['pix_x']/2, experiment['detector']['pix_y']/2
    nh, nv = experiment['detector']['n_pix_x'], experiment['detector']['n_pix_y']
    h, v = nh*dh/2, nv*dv/2
  
    beta_max, nbeta = experiment['angle_max'], data.shape[1]
    dbeta = beta_max/nbeta

    magn = D/Dsd
    dx, dy, dz = dh*magn, dh*magn, dv*magn
    nx, ny, nz = nv, nv, nv
    x, y, z = dx*nx/2, dy*ny/2, dz*nz/2


    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta,
                fourier = True)
    
    recon = reconstruction(lab, data, experiment['gpus'], save_path = experiment['save_path'], save = experiment['save'])

    print('fdk')

    return recon


