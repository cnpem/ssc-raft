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

def reconstruction_fdk( data, experiment):

    D, Dsd = experiment['z1'], experiment['z2']

    h, v = experiment['detector']['vis_x']/2, experiment['detector']['vis_y']/2
    nh, nv = experiment['detector']['n_pix_x'], experiment['detector']['n_pix_y']
    dh, dv = 2*h/(nh-1), 2*v/(nv-1)
  
    beta_max, nbeta = experiment['angle_max'], data.shape[1]
    dbeta = beta_max/nbeta

    magn = D/Dsd
    x, y, z = h*magn, h*magn, v*magn
    nx, ny, nz = nh, nh, nv
    dx, dy, dz = 2*x/(nx-1),  2*y/(ny-1), 2*z/(nz-1)


    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, dh = dh, dv = dv, nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, dbeta = dbeta , nbeta = nbeta,
                fourier = True)
    

    reconstruction(lab, data, experiment['gpus'], save_path = experiment['save_path'], save = True, vis = experiment['vis'])

    print('fdk')




def reconstruction( lab, proj, gpus, 
                    path_phantom = '', save_path = '', 
                    save = True, vis = True, comp = False):

    print("Reconstruction...")

    print("Projection shape  = "+ str(proj.shape))

    recon, filt = fdk_gpu(lab, proj, gpus)

    print("Reconstruction Shape  = "+ str(recon.shape))

    if save:
        print('Saving Results...')
        np.save(save_path+ "/recon_"+str(lab.nx)+"_"+ str(lab.nbeta)+".npy", recon)
        np.save(save_path+"/filter_"+str(lab.nx)+"_"+ str(lab.nbeta)+".npy", filt)

    if vis:
        print('Visualization...')
        phantom = np.load(path_phantom)
        visualization(lab, proj, filt, phantom, recon, len(gpus))
    
    if comp:
        print('Visualization...')
        phantom = np.load(path_phantom)
        vis_phantom(lab, proj, filt, phantom, recon, len(gpus))


def fdk_gpu(lab, proj, gpus):

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

    return recon, proj


def visualization(lab, cones, proj, recon, n_gpus, save_path):
    _, axs = plt.subplots(1,3)
    axs[0].imshow(proj[int(lab.nv/2), :,:])
    axs[1].imshow(proj[:, int(lab.nbeta/2), :])
    axs[2].imshow(proj[:,: ,int(lab.nh/2)])
    plt.savefig(save_path+'/filter'+str(lab.nx)+"_"+ str(lab.nbeta)+"_"+'.png')

    _, axs = plt.subplots(1,3)
    axs[0].imshow(recon[:, int(lab.ny/2), :])
    axs[1].imshow(recon[int(lab.nz/2), :,:])
    axs[2].imshow(recon[:, :,int(lab.nx/2)])
    plt.savefig(save_path+'/recon'+str(lab.nx)+"_"+ str(lab.nbeta)+"_"+'.png')

    images = []
    for k in range(int(lab.nbeta/2),int(lab.nbeta/2)+20):
        _, axs = plt.subplots(1,3)
        axs[0].imshow(cones[:, k, :])
        axs[1].imshow(proj[:, k, :])
        axs[3].imshow(recon[k%lab.nz, :, :])
        
        plt.savefig('fdkTESTE' + str(k) + '.png')
        plt.close()
        images.append(imageio.imread('fdkTESTE' + str(k) + '.png'))
        os.remove('fdkTESTE' + str(k) + '.png')          
    imageio.mimsave(save_path+'/fdk'+str(lab.nx)+"_"+ str(lab.nbeta)+"_"+str(int(n_gpus))+'.gif', images) 

def vis_phantom(lab, cones, proj, ph, recon, n_gpus, save_path):
    _, axs = plt.subplots(1,3)
    axs[0].imshow(proj[int(lab.nv/2), :,:])
    axs[1].imshow(proj[:, int(lab.nbeta/2), :])
    axs[2].imshow(proj[:,: ,int(lab.nh/2)])
    plt.savefig(save_path+'filter'+str(lab.nx)+"_"+ str(lab.nbeta)+"_"+'.png')

    _, axs = plt.subplots(1,3)
    axs[0].imshow(recon[:, int(lab.ny/2), :])
    axs[1].imshow(recon[int(lab.nz/2), :,:])
    axs[2].imshow(recon[:, :,int(lab.nx/2)])
    plt.savefig(save_path+'/recon'+str(lab.nx)+"_"+ str(lab.nbeta)+"_"+'.png')

    images = []
    for k in range(int(lab.nbeta/2),int(lab.nbeta/2)+20):
        _, axs = plt.subplots(1,4)
        axs[0].imshow(cones[:, k, :])
        axs[1].imshow(proj[:, k, :])
        axs[2].imshow(ph[k%lab.nz, :, :])
        axs[3].imshow(recon[k%lab.nz, :, :])
        
        plt.savefig('fdkTESTE' + str(k) + '.png')
        plt.close()
        images.append(imageio.imread('fdkTESTE' + str(k) + '.png'))
        os.remove('fdkTESTE' + str(k) + '.png')          
    imageio.mimsave(save_path+'/fdk'+str(lab.nx)+"_"+ str(lab.nbeta)+"_"+str(int(n_gpus))+'.gif', images)          
    

