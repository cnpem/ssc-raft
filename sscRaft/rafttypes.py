#We use the one encoding: utf8

# Ctypes =========
import ctypes
from ctypes import *
import ctypes.util

# General =========
import os
import sys
import numpy
import numpy as np
import json
import h5py
import time
from time import time
import warnings
import pathlib
import inspect
import matplotlib.pyplot as plt

# Multiprocessing =========
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory as SM
import multiprocessing as mp

# wiggle.py ============
import uuid
import SharedArray as sa
from scipy.optimize import minimize


# alignment.py =========

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

import skimage
import scipy
from skimage.registration import phase_cross_correlation
from skimage.transform import pyramid_gaussian
from skimage.transform import pyramid_reduce
from scipy.ndimage import center_of_mass
# ======================

from sscRaft import __version__

'''----------------------------------------------'''
import logging

console_log_level = 10

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)
console_handler.setLevel(console_log_level)

DEBUG = 10
'''----------------------------------------------'''

nthreads = multiprocessing.cpu_count()

# Load required libraies:

libstdcpp = ctypes.CDLL( ctypes.util.find_library( "stdc++" ), mode=ctypes.RTLD_GLOBAL )
# libblas   = ctypes.CDLL( ctypes.util.find_library( "blas" ), mode=ctypes.RTLD_GLOBAL )
# libfftw3  = ctypes.CDLL( ctypes.util.find_library( "fftw3" ), mode=ctypes.RTLD_GLOBAL )
# libfftw3_threads  = ctypes.CDLL( ctypes.util.find_library( "fftw3_threads" ), mode=ctypes.RTLD_GLOBAL )


_lib = "lib/libraft"

ext = '.so'

def load_library(lib,ext):
    _path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib + ext
    lib = ctypes.CDLL(_path)
    return lib

libraft  = load_library(_lib, ext)

#########################

#########################
#|       ssc-raft      |#
#| Function prototypes |#
#########################

######## Parallel Raft ##########

try:
    # EM Ray Tracing MultiGPU without semafaro
    libraft.get_tEM_RT_MultiGPU.argtypes = [
        ctypes.c_void_p, c_int, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    
    libraft.get_tEM_RT_MultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_eEMRT-')
    pass

try:
    libraft.get_eEM_RT_MultiGPU.argtypes = [
        ctypes.c_void_p, c_int, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]

    libraft.get_eEM_RT_MultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_tEMRT-')
    pass

try:
    # EM Frequency MultiGPU without semafaro
    libraft.get_tEM_FQ_MultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
        
    libraft.get_tEM_FQ_MultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_tEMFQ-')
    pass

try:
    # FBP 
    libraft.getBSTMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p
    ]
    
    libraft.getBSTMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_BST-')
    pass

try:
    # BST 
    libraft.getFBPMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p
    ]
    
    libraft.getFBPMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_FBP-')
    pass

try:
    libraft.findcentersino.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int
    ]

    libraft.findcentersino.restype = ctypes.c_int    
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_CENTERSINO-')
    pass

######## Raft - Rings ##########
try:
    libraft.getTitarenkoRingsMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,  
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_float, ctypes.c_int
    ]
    
    libraft.getTitarenkoRingsMultiGPU.restype  = None
    
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_TITARENKO_RINGS-')
    pass

######## Raft - Flat/Dark Correction ##########
try:
    libraft.getBackgroundCorrectionMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int
    ]
    
    libraft.getBackgroundCorrectionMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_BACKGROUND_CORRECTION-')
    pass

######## Raft - Stitching Offset 360 ##########
try:
    libraft.getOffsetStitch360GPU.argtypes = [
        ctypes.c_int, ctypes.c_void_p,  
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    
    libraft.getOffsetStitch360GPU.restype  = ctypes.c_int
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_OFFSET_STITCHING360-')
    pass

######## Raft - Stitch 360 to 180 ##########
try:
    libraft.stitch360To180MultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_int
    ]
    
    libraft.stitch360To180MultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_STITCH_360TO180-')
    pass

######## Raft - Phase Retrieval Paganin method and similar methods ##########
try:
    libraft.getPhaseMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    
    libraft.getPhaseMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_PHASE_RETRIEVAL-')
    pass

######## Raft - Parallel Radon ##########
try:
    libraft.getRadonRTMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_float, ctypes.c_float
    ]

    libraft.getRadonRTMultiGPU.restype  = None 
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_RADON_RT-')
    pass

######## Raft - FDK ##########
class Lab(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float),
                ("dx", ctypes.c_float), ("dy", ctypes.c_float),("dz", ctypes.c_float),
                ("nx", ctypes.c_int), ("ny", ctypes.c_int), ("nz", ctypes.c_int),
                ("h", ctypes.c_float), ("v", ctypes.c_float),
                ("dh", ctypes.c_float), ("dv", ctypes.c_float),
                ("nh", ctypes.c_int), ("nv", ctypes.c_int),
                ("D", ctypes.c_float), ("Dsd", ctypes.c_float),
                ("beta_max", ctypes.c_float),
                ("dbeta", ctypes.c_float),
                ("nbeta", ctypes.c_int),
                ("fourier", ctypes.c_int),
                ("filter_type", ctypes.c_int),
                ("reg", ctypes.c_float),
                ("is_slice", ctypes.c_int),
                ("slice_recon_start", ctypes.c_int),
                ("slice_recon_end", ctypes.c_int),
                ("slice_tomo_start", ctypes.c_int),
                ("slice_tomo_end", ctypes.c_int),
                ("nph", ctypes.c_int),
                ("padh", ctypes.c_int),
                ("energy", ctypes.c_float)
                ]

try:
    libraft.gpu_fdk.argtypes = [
        Lab, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
    ]
    libraft.getBackgroundCorrectionMultiGPU.restype  = None
except:
    print('Cannot find C/CUDA library: -.RAFT_FDK-')
    pass

## Transmission Expectation-Maximization (tEMRT - conebeam):
class Lab_EM(ctypes.Structure):
    _fields_ = [
        ("Lx", ctypes.c_float), ("Ly", ctypes.c_float), ("Lz", ctypes.c_float),
        ("x0", ctypes.c_float), ("y0", ctypes.c_float), ("z0", ctypes.c_float),
        ("nx", ctypes.c_int), ("ny", ctypes.c_int), ("nz", ctypes.c_int),
        ("sx", ctypes.c_float), ("sy", ctypes.c_float), ("sz", ctypes.c_float),
        ("nbeta", ctypes.c_int),
        ("ndetc", ctypes.c_int),
        ("n_ray_points", ctypes.c_int)]

lib_cone_tEM  = load_library(_lib, ext)

try:
    libraft.conebeam_tEM_gpu.argtypes = [
        Lab_EM, # struct Lab lab.
        ctypes.c_void_p, # float *flat.
        ctypes.c_void_p, # float *px.
        ctypes.c_void_p, # float *py.
        ctypes.c_void_p, # float *pz.
        ctypes.c_void_p, # float *angles.
        ctypes.c_void_p, # float *recon.
        ctypes.c_void_p, # float *tomo.
        ctypes.c_int, # int ngpus.
        ctypes.c_int, # int niter.
        ctypes.c_float, # float tv.
        ctypes.c_float] # float max_val.
    libraft.conebeam_tEM_gpu.restype = ctypes.c_int
except:
    raise NotImplementedError()

######## Raft - Conebeam Radon by Ray Tracing ##########
class Lab_CB(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float),
                ("x0", ctypes.c_float),
                ("y0", ctypes.c_float),
                ("z0", ctypes.c_float),
                ("nx", ctypes.c_int),
                ("ny", ctypes.c_int),
                ("nz", ctypes.c_int),
                ("sx", ctypes.c_float),
                ("sy", ctypes.c_float),
                ("sz", ctypes.c_float),
                ("nbeta", ctypes.c_int),
                ("n_detector", ctypes.c_int),
                ("n_ray_points",  ctypes.c_int)]
    
try:
    libraft.cbradon.argtypes = [
        Lab_CB, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int
    ]
    libraft.cbradon.restype  = ctypes.c_int
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_CONEBEAM_RADON_RT-')
    pass


#########################
#|      ssc-raft       |#
#|      Functions      |#
#########################

def CNICE(darray,dtype=numpy.float32):
        if darray.dtype != dtype:
                return numpy.ascontiguousarray(darray.astype(dtype))
        elif darray.flags['C_CONTIGUOUS'] == False:
                return numpy.ascontiguousarray(darray)
        else:
                return darray

VERBOSE = False
def dprint(*x):
        if VERBOSE:
                print(*x)

def nice(f): # scientific notation + 2 decimals
    return "{:.2e}".format(f)

def FilterNumber(mfilter):
    if mfilter.lower() == 'none':
        return 0
    elif mfilter.lower() == 'gaussian':
        return 1
    elif mfilter.lower() == 'lorentz':
        return 2
    elif mfilter.lower() == 'cosine':
        return 3
    elif mfilter.lower() == 'rectangle':
        return 4
    elif mfilter.lower() == 'hann':
        return 5
    elif mfilter.lower() == 'hamming':
        return 6
    elif mfilter.lower() == 'ramp':
        return 7
    else:
        return 6

def PhaseMethodNumber(mfilter):
    if mfilter.lower() == 'paganin':
        return 0
    elif mfilter.lower() == 'tomopy':
        return 1
    elif mfilter.lower() == 'v0':
        return 2
    else:
        return 0

def setInterpolation(name):
    """ Set interpolation 

    Args:
        name (str): string name for the interpolation

    Returns:
        (int): Value defining the interpolation name

    Options:
        name (str): \'nearest\' and \'bilinear\'

    """
    if name.lower() == 'nearest':
        return 0
    elif name.lower() == 'bilinear':
        return 1
    else:
        logger.warning(f'Interpolation invalid. Using default \'nearest\' interpolation.')
        return 0
        
def set_precision(precision):
    """Select datatype of numpy array.

    Args:
        precision (str): precision name

    Returns:
        (int): Integer value defining the numpy datatype
        (type): The numpy datatype

    Raises:
        Invalid datatype

    Options:
        precision (str): \'float32\', \'uint16\', \'uint8\'
    """
    if precision == 'float32':
        datatype = 5
        Ndatatype = numpy.float32
    elif precision == 'uint16':
        datatype = 2
        Ndatatype = numpy.uint16
    elif precision == 'uint8':
        datatype = 1
        Ndatatype = numpy.uint8
    else:
        logger.error(f'Invalid datatype:{precision}. Options: `float32`, `uint16` and `uint8`.')
        raise ValueError(f'Invalid datatype:{precision}. Options: `float32`, `uint16` and `uint8`.')

    return datatype, Ndatatype

def power_of_2_padding(size,pad):
    return int((pow(2, numpy.ceil(numpy.log2(size + 2 * pad))) - size) * 0.5)
       
def crop_circle(data, z1, zt, pixeldet, npixels = 0):
    r"""Function to crop a circle of radius :math:`r` on the xy-plane of a 3D data.
        
        - r = (radius - npixels * pixel_size);
        - radius = (pixel_size * dimension) / 2;
        - pixel_size = pixel_detector * magnitude;
        - magnitude = z1/zt
        - zt = z1+z2

    Args:
        data (ndarray): 3D data
        z1 (float): Sample-source distance in meters
        zt (float): Detector-source distance in meters (z1+z2)
        pixeldet (float): Detector pixel size in meters 
        npixels (int, optional): How many pixels to subtract from radius to reduce cropped circle [Default: 0]

    Returns:
        (ndarray): 3D cropped data
    """
    dimx = data.shape[-1]
    dimy = data.shape[-2]

    pixel_size = pixeldet * ( z1 / zt )

    diameterx = dimx * pixel_size
    radiusx = diameterx / 2.0

    diametery =  dimy * pixel_size
    radiusy = diametery / 2.0

    x = numpy.linspace(-radiusx,radiusx,dimx)
    y = numpy.linspace(-radiusy,radiusy,dimy)

    Xm, Ym = numpy.meshgrid(x,y)
    radius = max(radiusx,radiusy)
    r = radius - npixels * pixel_size

    mask = (Xm)**2 + (Ym)**2  <= r**2

    for i in range(data.shape[0]):
            data[i,:,:] = numpy.where(mask,data[i,:,:],0)

    return data

def crop_ellipse(data, pixeldet, magn = 1, npixelsx = 0, npixelsy = 0):
    r"""Function to crop an ellipse of radius :math:`a` and :math:`b` on the xy-plane of a 3D data.

    Args:
        data (ndarray): 3D data
        pixeldet (float tuple): Detector pixel size (x,y) in x- and y- directions in meters 
        magn (float): Magnification related to beam geometry (z1+z2)/z1 [Default: 1, parallel geometry]
        npixelsx (int, optional): How many pixels to subtract from radius ``a`` (x-direction) to reduce cropped circle [Default: 0]
        npixelsy (int, optional): How many pixels to subtract from radius ``b`` (y-direction) to reduce cropped circle [Default: 0]

    Returns:
        (ndarray): 3D cropped data
    """
    dimx = data.shape[-1]
    dimy = data.shape[-2]

    pixel_sizex = pixeldet[0] * ( 1 / magn )
    pixel_sizey = pixeldet[1] * ( 1 / magn )

    diameterx = dimx * pixel_sizex
    a = diameterx / 2.0 - npixelsy * pixel_sizey

    diametery =  dimy * pixel_sizey
    b = diametery / 2.0 - npixelsx * pixel_sizex

    x = numpy.linspace(-a,a,dimx)
    y = numpy.linspace(-b,b,dimy)

    Xm, Ym = numpy.meshgrid(x,y)

    mask = (Xm/a)**2 + (Ym/b)**2  <= 1

    for i in range(data.shape[0]):
            data[i,:,:] = numpy.where(mask,data[i,:,:],0)
            
    return data

if __name__ == "__main__":
   pass

