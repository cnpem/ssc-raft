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

nthreads = multiprocessing.cpu_count()

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

#==========================================================
# Definitions and Constants
#==========================================================
LIGHT_VELOCITY = 299792458                         # [m/s]
PLANK_CONSTANT = 4.135667662E-15                   # [eV.s]
CONST          = PLANK_CONSTANT * LIGHT_VELOCITY   # [eV.m]
#==========================================================

############# Basic functions ##############

def ReconMethod(method):
    if method.lower() == 'none':
        return 0
    elif method.lower() == 'fbp':
        return 1
    elif method.lower() == 'bst':
        return 2
    elif method.lower() == 'eEMRT':
        return 3
    elif method.lower() == 'tEMRT':
        return 4
    elif method.lower() == 'tEMFQ':
        return 5
    elif method.lower() == 'fdk':
        return 6
    else:
        return 1

def RingsMethod(method):
    if method.lower() == 'none':
        return 0
    elif method.lower() == 'all stripes':
        return 1
    elif method.lower() == 'all stripes multiaxis':
        return 2
    elif method.lower() == 'titarenko':
        return 3
    else:
        return 1

def AlignMethod(method):
    if method.lower() == 'none':
        return 0
    elif method.lower() == 'rot axis recon':
        return 1
    elif method.lower() == 'rot axis':
        return 2
    elif method.lower() == 'excentric':
        return 3
    else:
        return 1
    
def BckCorrMethod(method):
    if method.lower() == 'none':
        return 0
    elif method.lower() == 'classic':
        return 1
    elif method.lower() == 'excentric':
        return 2
    else:
        return 1
    
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

def ContrastFilterNumber(mfilter):
    if mfilter.lower() == 'none':
        return 0
    elif mfilter.lower() == 'paganin':
        return 1
    elif mfilter.lower() == 'paganin_slices':
        return 2
    elif mfilter.lower() == 'contrast':
        return 3
    else:
        return 1

def PaddMode(pmode):
    if pmode.lower() == 'none':
        return 0
    elif pmode.lower() == 'zero':
        return 1
    elif pmode.lower() == 'edge':
        return 2
    elif pmode.lower() == 'ones':
        return 3
    else:
        return 2
    
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
    elif name.lower() == 'none':
        return -1
    else:
        logger.warning(f'Interpolation invalid. Using default \'nearest\' interpolation.')
        return 0

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

def sanitize_angles(angles: numpy.ndarray) -> numpy.ndarray:
    ''' Set angles to start at 0 (zero) radians. 
    '''
    angles = angles - min(angles)

    return angles

def get_excentric_obj_dimensions(nx,ny):
     return 2 * nx, 2 * ny

def get_padding_size(n,pad):
     return int( (pad * n) / 100 )

def set_input_axis_order(order: str = 'slices_angles_lenght'):
    # Default is 'slices_angles_lenght'

    if order.lower() == 'angles_slices_lenght':
        return 0
    elif order.lower() == 'slices_angles_lenght':
        return 1
    elif order.lower() == 'transpose_lenght':
        return 2
    else:
        message = f'The axis input parameter \'{order}\' does not exist. Setting \'slices_angles_lenght\' as default.'
        logger.warning(message)
        return 1 
    
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

def crop_circle_simple(data, pixel_size, npixels = 0):
    r"""Function to crop a circle of radius :math:`r` on the xy-plane of a 3D data.
        
        - r = (radius - npixels * pixel_size);


    Args:
        data (ndarray): 3D data
        pixel_size (float): Pixel size in meters 
        npixels (int, optional): How many pixels to subtract from radius to reduce cropped circle [Default: 0]

    Returns:
        (ndarray): 3D cropped data
    """
    dimx = data.shape[-1]
    dimy = data.shape[-2]

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
