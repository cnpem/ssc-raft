from ....rafttypes import *
import numpy as np
import ctypes
import ctypes.util
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t

def set_experiment( x, y, z, 
                    dx, dy, dz, 
                    nx, ny, nz, 
                    h, v, 
                    dh, dv, 
                    nh, nv,
                    D, Dsd, 
                    beta_max, 
                    dbeta, 
                    nbeta,
                    fourier, 
                    filter, 
                    regularization,
                    start_slice, end_slice, nslices):

    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, 
                dh = dh, dv = dv, 
                nh = nh, nv = nv, 
                D = D, Dsd = Dsd, 
                beta_max = beta_max, 
                dbeta = dbeta, 
                nbeta = nbeta,
                fourier = fourier, 
                filter_type = filter, 
                reg = regularization,
                slice0 = start_slice, slice1 = end_slice, nslices = nslices)

    return lab


def fdk(tomogram: np.ndarray, dic: dict = {}) -> np.ndarray:
    """Computes the Reconstruction of a Conical Sinogram using the Filtered Backprojection method for conical rays(FDK).
    GPU function.

    Args:
        data (ndarray): Cone beam projection tomogram. The axes are [slices, angles, lenght].
        dic (dictionary): Dictionary with the experiment info.

    Returns:
        (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [z, y, x].

    Dictionary parameters:
        *``dic['gpu']`` (ndarray): List of gpus for processing. Defaults to [0].
        *``dic['z1[m]']`` (float): Source-sample distance in meters. Defaults to 500e-3.
        *``dic['z1+z2[m]']`` (float): Source-detector distance in meters. Defaults to 1.0.
        *``dic['detectorPixel[m]']`` (float): Detector pixel size in meters. Defaults to 1.44e-6.
        *``dic['reconSize']`` (int): Reconstruction dimension. Defaults to data shape[-1].
        *``dic['filter']`` (str,optional): Type of filter for reconstruction. 
        *``dic['angles']`` (list): list of angles.
        *``dic['filter']`` (str,optional): Type of filter for reconstruction. 
        Options = ('none','gaussian','lorentz','cosine','rectangle','hann','hamming','ramp'). Default is 'hamming'.
        *``dic['regularization']`` (float,optional): Type of filter for reconstruction, small values. Default is 1.
    """

    # recon = data 
    regularization = dic['regularization']

    # try:
    #     padding = int(dic['padding'])
    # except:
    #     logger.warning(f'Set default padding size of {padding}.')

    # padding = 0

    # proj = np.zeros((tomogram.shape[0], tomogram.shape[1], tomogram.shape[2] + 2 * padding))

    # proj[:,:,padding:tomogram.shape[2] + padding] = tomogram

    if len(tomogram.shape) == 2:
            nslices = 1
    elif len(tomogram.shape) == 3:
            nslices = tomogram.shape[0]
    else:
        logger.error(f'Data has wrong shape = {tomogram.shape}. It needs to be 2D or 3D.')
        sys.exit(1)

    nrays   = tomogram.shape[-1]
    nangles = tomogram.shape[-2]

    try:
        angles     = dic['angles']
    except Exception as e:
        logger.error(f'Missing angles list entry from dictionary!')
        logger.exception(e)
    
    try:
        start_slice = dic['slices'][0]
        end_slice   = dic['slices'][1]
        logger.info(f'Reconstruct slices {start_slice} to {end_slice}.')
    except:
        start_slice = 0
        end_slice   = nslices   

    blockslices = end_slice - start_slice

    if blockslices > nslices:
        logger.error(f'Trying to reconstruct more slices than provided by the tomogram data.')
        sys.exit(1)       

    Dd, Dsd = dic['z1[m]'], dic['z1+z2[m]']

    dh, dv = dic['detectorPixel[m]'], dic['detectorPixel[m]']
    nh, nv = int(nrays), int(blockslices)
    h, v   = nh*dh/2, nslices*dv/2
    # h, v   = nh*dh/2, nv*dv/2

    nbeta  = len(angles)

    if nbeta != nangles: 
        logger.error(f'Number of projection do not match: size of angles list ({nbeta}) is different from the number of projections ({nangles}).')
        logger.error(f'Finishing run...')
        sys.exit()

    beta_max = angles[nbeta - 1]
    dbeta    = angles[1] - angles[0]

    # beta_max, nbeta = 2*np.pi, int(tomogram.shape[1])
    # dbeta           = beta_max/nbeta

    magn       = Dd/Dsd
    dx, dy, dz = dh*magn, dh*magn, dv*magn

    try:
        reconsize = dic['reconSize']
        
        if len(dic['reconSize']) == 1:
            nx, ny, nz = int(reconsize[0]), int(reconsize[0]), int(blockslices)
        elif len(dic['reconSize']) == 2:
            nx, ny, nz = int(reconsize[0]), int(reconsize[1]), int(blockslices)
        elif len(dic['reconSize']) > 2:
            nx, ny, nz = int(reconsize[0]), int(reconsize[1]), int(reconsize[2])
    except:
        nx, ny, nz = int(nrays), int(nrays), int(blockslices)
        logger.info(f'Set default reconstruction size ({nz},{ny},{nx}) = (slices,angles,rays).')
    

    # x, y, z    = dx*nx/2, dy*ny/2, dz*nz/2
    x, y, z    = dx*nx/2, dy*ny/2, dz*nslices/2

    fourier    = dic['fourier']
    filtername = dic['filter']
    filter     = FilterNumber(dic['filter'])

    logger.info(f'FDK filter: {filtername}({filter})')

    if filter == 1 or filter == 2 or filter == 4:
        logger.info(f'FDK filter regularization: {regularization}')     

    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, 
                dh = dh, dv = dv, 
                nh = nh, nv = nv, 
                D = Dd, Dsd = Dsd, 
                beta_max = beta_max, 
                dbeta = dbeta , 
                nbeta = nbeta,
                fourier = fourier, 
                filter_type = filter, 
                reg = regularization,
                slice0 = start_slice, slice1 = end_slice, nslices = nslices)

    time = np.zeros(2)
    time = np.ascontiguousarray(time.astype(np.float64))
    time_p = time.ctypes.data_as(ctypes.c_void_p)

    gpus = np.array(dic['gpu'])
    ndev = len(gpus)
    gpus = np.ascontiguousarray(gpus.astype(np.intc))
    gpus_p = gpus.ctypes.data_as(ctypes.c_void_p)

    angles = np.ascontiguousarray(angles.astype(np.float32))
    angles_p = angles.ctypes.data_as(ctypes.c_void_p)

    logger.info(f'Tomogram data shape: {tomogram.shape} = (slices,angles,rays).')

    proj = np.ascontiguousarray(tomogram.astype(np.float32))
    proj_p = proj.ctypes.data_as(ctypes.c_void_p)

    recon = np.zeros((lab.nz, lab.ny, lab.nx))
    recon = np.ascontiguousarray(recon.astype(np.float32))
    recon_p = recon.ctypes.data_as(ctypes.c_void_p)

    libraft.gpu_fdk(lab, recon_p, proj_p, angles_p, gpus_p, int(ndev), time_p)

    # print("Time of execution: \n \n"+"Phantom size"+str(lab.nx)+"\n Nbeta"+str(lab.nbeta)+"\n ngpus"+str(ndev)+"\n"+str(time))

    return recon
