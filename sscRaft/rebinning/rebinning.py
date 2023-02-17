#!/usr/bin/env python3

from ..rafttypes import *

import numpy
import numpy as np
import ctypes
from ctypes import * 
from ..rafttypes import *

def SetDic(dic, paramname, deff):
        try:
                dic[paramname]
        except:
                logger.info(f'Using default - {paramname}:{deff}')
                dic[paramname] = deff

def SetDictionary(dic,param,default):
    for ind in range(len(param)):
        SetDic(dic,param[ind], default[ind])

def rebinning_gpu(conetomo, dic, **kwargs):
    
    z1x, z1y = dic['z1']
    z2x, z2y = dic['z2']
    d1x, d1y = dic['d1']
    d2x, d2y = dic['d2']
    px, py = dic['DetPixelSize']

    gpus = numpy.array(dic['gpus'])
    ngpus  = gpus.shape[0]
    gpus   = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_p = gpus.ctypes.data_as(ctypes.c_void_p)

    tomo = numpy.zeros(conetomo.shape)
    tomo   = numpy.ascontiguousarray(tomo.astype(numpy.float32))
    tomo_p = tomo.ctypes.data_as(ctypes.c_void_p)

    param = numpy.array([z1x,z1y,z2x,z2y,d1x,d1y,d2x,d2y,px,py])
    param   = numpy.ascontiguousarray(param.astype(numpy.float32))
    param_p = param.ctypes.data_as(ctypes.c_void_p)

    sizex = conetomo.shape[-1]
    sizey = conetomo.shape[-2]
    sizez = conetomo.shape[ 0]

    blocksize = sizez

    if blocksize > sizez:
       print('Blocksize = {0} bigger than projections = {1}. Blocksize needs to be smaller or equal than projections size'.format(blocksize,sizez))
       quit()

    volumesize = numpy.array([sizex,sizey,sizez, blocksize])
    volumesize   = numpy.ascontiguousarray(volumesize.astype(numpy.int64))
    volumesize_p = volumesize.ctypes.data_as(ctypes.c_void_p)
    
    conetomo   = numpy.ascontiguousarray(conetomo.astype(numpy.float32))
    conetomo_p = conetomo.ctypes.data_as(ctypes.c_void_p)

    
    libraft.GPUrebinning(conetomo_p, tomo_p, param_p, volumesize_p, gpus_p, ngpus)

    del(conetomo_p)
    del(param_p)
    del(volumesize_p)
    del(gpus_p)

    return tomo

def rebinning_cpu(conetomo, dic, **kwargs):
    
    z1x, z1y = dic['z1']
    z2x, z2y = dic['z2']
    d1x, d1y = dic['d1']
    d2x, d2y = dic['d2']
    px, py = dic['DetPixelSize']

    tomo = numpy.zeros(conetomo.shape)
    tomo   = numpy.ascontiguousarray(tomo.astype(numpy.float32))
    tomo_p = tomo.ctypes.data_as(ctypes.c_void_p)

    param = numpy.array([z1x,z1y,z2x,z2y,d1x,d1y,d2x,d2y,px,py])
    param   = numpy.ascontiguousarray(param.astype(numpy.float32))
    param_p = param.ctypes.data_as(ctypes.c_void_p)

    sizex = conetomo.shape[-1]
    sizey = conetomo.shape[-2]
    sizez = conetomo.shape[ 0]

    blocksize = sizez

    if blocksize > sizez:
       logger.error('Blocksize = {0} bigger than projections = {1}. Blocksize needs to be smaller or equal than projections size'.format(blocksize,sizez))
       quit()

    volumesize = numpy.array([sizex,sizey,sizez, blocksize])
    volumesize   = numpy.ascontiguousarray(volumesize.astype(numpy.int64))
    volumesize_p = volumesize.ctypes.data_as(ctypes.c_void_p)
    
    conetomo   = numpy.ascontiguousarray(conetomo.astype(numpy.float32))
    conetomo_p = conetomo.ctypes.data_as(ctypes.c_void_p)

    libraft.CPUrebinning(conetomo_p, tomo_p, param_p, volumesize_p)

    del(conetomo_p)
    del(param_p)
    del(volumesize_p)

    return tomo

def rebinning_python(conesino, z1, z2, detector_size=(1, 1), pdetector_size=(1, 1), PONI = (0,0), rot = (0,0), pha = (0,0)):
    """Computes the Conebeam Rebinning to Parallel rays of a tomogram (3D).

    Args:
        conesino (ndarray): Cone beam tomogram. The axes are [angles, slices, X].
        z1 (float): Distance from source to object.
        z2 (float): Distance from object to detector.
        detector_size (tuple, optional): Cone beam detector sizes (Dx = 1, Dy = 1), where horizontal size is [-Dx,Dx] and vertical size is [-Dy,Dy].
        pdetector_size(tuple, optional): Parallel beam detector sizes (Lx = Dx, Ly = Dy), where horizontal size is [-Lx,Lx] and vertical size is [-Ly,Ly].
        PONI (tuple, optional): Poni (centra ray at detector) shift (cx = 0, cy = 0).
        rot (tuple, optional): Rotation center shift from poni (rx = 0, ry = 0).
        pha (tuple, optional): Phantom shift from rotation center (rx = 0, ry = 0).

    Returns:
        (ndarray): Rebinned Parallel beam tomogram (3D). The axes are [angles, slices, X].
    """

    ntheta, nb, na = nbeta, nr, nt  = conesino.shape

    tsize, rsize = detector_size
    rm, tm = rsize, tsize

    asize, bsize = pdetector_size
    bm, am = bsize, asize

    t = np.linspace(-tm, tm, nt, endpoint=False)
    r = np.linspace(-rm, rm, nr, endpoint=False)
    bt = np.linspace(0, 360*np.pi/180, nbeta, endpoint=False)
    dbeta = bt[1] - bt[0]
    print(nbeta, dbeta, 2*np.pi/(nbeta), 2*np.pi/(nbeta-1))
    print(nt, 2*tm / nt, 2*tm / (nt -1), t[1] - t[0]) 
    print(nr, 2*rm / nr, 2*rm / (nr -1), r[1] - r[0])

    aid = lambda a: np.around((a + (am))/2/(am)*(na)).astype(int)
    bid = lambda b: np.around((b + (bm))/2/(bm)*(nb)).astype(int)
    thetaid = lambda beta: np.around((beta/(dbeta))).astype(int)%nbeta
    
    parsino = np.zeros((ntheta, nb, na))

    beta, rr, tt = np.meshgrid(bt, r, t, indexing='ij') 

    ib                       = bid( ( rr * (z1 + z2) ) / np.sqrt((z1+z2)**2 + rr**2) ) 
    ia                       = aid( ( tt * (z1 + z2) ) / np.sqrt((z1+z2)**2 + tt**2) ) 
    itheta                   = thetaid( beta + np.arctan( tt / (z1+z2) ) )
    ib[ib < 0]               = 0
    ib[ib > nb - 1]          = nb - 1
    ia[ia < 0]               = 0
    ia[ia > na - 1]          = na - 1
    parsino[itheta, ib, ia]  = conesino

    return parsino

def rebinning_projection(conesino, z1, z2, detector_size=(1, 1), pdetector_size=(1, 1), PONI = (0,0), rot = (0,0), pha = (0,0)):
    """Computes the Conebeam Rebinning to Parallel rays of a Projection (2D).

    Args:
        conesino (ndarray): Cone beam tomograsm. The axes are [slices, X].
        z1 (float): Distance from source to object.
        z2 (float): Distance from object to detector.
        detector_size (tuple, optional): Cone beam detector sizes (Dx = 1, Dy = 1), where horizontal size is [-Dx,Dx] and vertical size is [-Dy,Dy].
        pdetector_size(tuple, optional): Parallel beam detector sizes (Lx = Dx, Ly = Dy), where horizontal size is [-Lx,Lx] and vertical size is [-Ly,Ly].
        PONI (tuple, optional): Poni (centra ray at detector) shift (cx = 0, cy = 0).
        rot (tuple, optional): Rotation center shift from poni (rx = 0, ry = 0).
        pha (tuple, optional): Phantom shift from rotation center (rx = 0, ry = 0).

    Returns:
        (ndarray): Rebinned Parallel beam tomogram (3D). The axes are [slices, X].
    """

    nb, na = nr, nt  = conesino.shape

    ssize, rsize = detector_size
    rm, tm = ssize, rsize

    bsize, asize = pdetector_size
    bm, am = bsize, asize

    t = np.linspace(-tm, tm, nt)
    r = np.linspace(-rm, rm, nr)

    aid = lambda a: np.around((a + (am))/2/(am)*(na - 1)).astype(int)
    bid = lambda b: np.around((b + (bm))/2/(bm)*(nb - 1)).astype(int)

    parsino = np.zeros((nb, na))

    rr, tt = np.meshgrid(r, t, indexing='ij') 

    ib              = bid( ( ( (rr )*(z1 + z2) ) / np.sqrt((z1+z2)**2 + (rr )**2) ) )
    ia              = aid( ( ( (tt )*(z1 + z2) ) / np.sqrt((z1+z2)**2 + (tt )**2) ) )
    ib[ib < 0]      = 0
    ib[ib > nb - 1] = nb - 1
    ia[ia < 0]      = 0
    ia[ia > na - 1] = na - 1
    parsino[ib, ia] = conesino

    return parsino

def conebeam_rebinning_to_parallel(conetomo, dic, **kwargs):
    """Computes the Conebeam Rebinning to Parallel rays of a tomogram (3D) or a projection (2D).

    Args:
        conesino (ndarray): Cone beam tomogram or projection. The axes are [angles, slices, X] (3D) or [slices, X] (2D).
        dic (dictionary): Dictionary to rebinning parameters.

    Returns:
        (ndarray): Rebinned Parallel beam tomogram (2D) or projection (3D). The axes are [angles, slices, X] (3D) or [slices, X] (2D).
    
    * CPU/GPU functions
    
    The dictionary (dic) parameters are:

    * ``dic['gpus']``: List of GPU devices used for computation 
    * ``dic['Distances']``: Tuple of source/sample and sample/detector, respectively (z1 = 2,z2 = 1)
    * ``dic['Poni']``: Tuple PONI (point of incidence) of central ray at detector (cx = 0,cy = 0)
    * ``dic['ShiftPhantom']``: Tuple of phantom shift (sx = 0,sy = 0) 
    * ``dic['ShiftRotation']``: Tuple of rotation center shift (rx = 0,ry = 0)
    * ``dic['DetectorSize']``: Tuple of detector size (Dx = 1,Dy = 1), where the size interval is [-Dx,Dx], [-Dy,Dy]
    * ``dic['ParDectSize']``: Tuple of detector size (Lx = 1,Ly = 1), where the size interval is [-Lx,Lx], [-Ly,Ly]
    * ``dic['PixelSize']``: Tuple of pixel sizes (px = Dx/Nx,py = Dy/Ny). Nx = nimages, Ny = nslices.
    * ``dic['Type']``: String (``cpu``,``gpu``,``py``) of function type - cpu, gpu, python, respectively - used to compute tomogram (3D). Defauts to ``cpu``.

    """
    dimension = len(conetomo.shape)

    if dimension == 2:
        nimages = conetomo.shape[1]
        nslices = conetomo.shape[0]
        
        # Set default values for dictionary 'dic' if it is NOT defined
        params = ('z1','z2','d1','d2','DetPixelSize')
        defaut = ((1,1),(0,0),(1,1),(0,0),(1/nimages,1/nslices))
        SetDictionary(dic,params,defaut)

        tomo = rebinning_projection(conetomo, dic['d1'][0], dic['d2'][0], (1,1), (1,1), (0,0), (0,0), (0,0))

        return tomo

    elif dimension == 3:
        nimages = conetomo.shape[2]
        nslices = conetomo.shape[1]
        
        # Set default values for dictionary 'dic' if it is NOT defined
        params = ('gpus','z1','z2','d1','d2','DetPixelSize', 'Type')
        defaut = ([0],(1,1),(0,0),(1,1),(0,0),(1/nimages,1/nslices),'cpu')
        SetDictionary(dic,params,defaut)

        print('Zs:', dic['d1'][0], dic['d2'][0])
        if dic['Type'] == 'cpu':
            tomo = rebinning_cpu(conetomo, dic, **kwargs)
            return tomo
        elif dic['Type'] == 'gpu':
            tomo = rebinning_gpu(conetomo, dic, **kwargs)
            return tomo
        elif dic['Type'] == 'py':
            tomo = rebinning_python(conetomo, dic['d1'][0], dic['d2'][0], (1,1), (1,1), (0,0), (0,0), (0,0))
            return tomo
        else:
            type = dic['Type']
            logger.error(f'Error! dictionary parameter Type = {type} is not defined')
    else:
        logger.error(f'Error! Dimension {dimension} of input ndarray is not suported. Enter a 2D or 3D input ndarray.')
    
    
