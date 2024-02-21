#We use the one encoding: utf8
import ctypes
from ctypes import *
import ctypes.util
import multiprocessing
import os
import sys
import numpy
import json
import h5py

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
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_float, ctypes.c_int
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
    libraft.getRingsMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_float, ctypes.c_int
    ]
    
    libraft.getRingsMultiGPU.restype  = None
    
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_RINGS-')
    pass

######## Raft - Flat/Dark Correction ##########
try:
    libraft.getFlatDarkMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int
    ]
    
    libraft.getFlatDarkMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_FLAT_DARK_CORRECTION-')
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

######## Conical Raft ##########
## FDK:
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
        libraft.gpu_fdk.argtypes = [Lab, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

except:
    print('-.RAFT_CONICAL-')
    pass

## Transmission Expectation-Maximization (TEM):
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
    conebeam_tEM_gpu = lib_cone_tEM.conebeam_tEM_gpu
    conebeam_tEM_gpu.argtypes = [
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
    conebeam_tEM_gpu.restype = ctypes.c_int
except:
    raise NotImplementedError()



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

def PhaseFilterNumber(mfilter):
    if mfilter.lower() == 'none':
            return 0
    elif mfilter.lower() == 'paganin':
            return 1
    elif mfilter.lower() == 'bronnikov':
            return 2
    elif mfilter.lower() == 'born':
            return 3
    elif mfilter.lower() == 'rytov':
            return 4
    else:
            return 0

def setInterpolation(name):
    """ Set interpolation 

    Args:
        name (str): string name for the interpolation

    Returns:
        (int): Value defining the interpolation name

    Options:
        name: \'nearest\' and \'bilinear\'

    """
    if name.lower() == 'nearest':
        return 0
    elif name.lower() == 'bilinear':
        return 1
    else:
        logger.warning(f'Interpolation invalid. Using default \'nearest\' interpolation.')
        return 0
        

def Bin(img,n=2):
        if n <= 1:
                return img
        elif n%2 != 0:
                if img.shape[0]%n != 0 or img.shape[1]%n != 0:
                        print('Warning: Non divisible binning is subsampling.')
                        return img[0::n,0::n]
                else:
                        img2 = numpy.zeros((img.shape[0]//n,img.shape[1]//n),dtype=img.dtype)
                        for j in range(n):
                                for i in range(n):
                                        img2 += img[j::n,i::n]
                        return img2/(n*n)
        else:
                return Bin(0.25*(img[0::2,0::2] + img[1::2,0::2] + img[0::2,1::2] + img[1::2,1::2]),n//2)


def SetDic(dic, paramname, deff):
        try:
            dic[paramname]
            
            if type(dic[paramname]) == list:
                for i in range(len(deff)):
                    try:
                        dic[paramname][i] 
                    except:
                        value = deff[i]
                        logger.info(f'Using default - {paramname}:{value}')
                        dic[paramname][i] = value

        except:
            logger.info(f'Using default - {paramname}: {deff}.')
            dic[paramname] = deff

def SetDictionary(dic,param,default):
    for ind in range(len(param)):
        SetDic(dic,param[ind], default[ind])

def Metadata_hdf5(outputFileHDF5, dic, software, version):
    """ Function to save metadata from a dictionary, and the name of the softare used and its version
    on a HDF5 file. The parameters names will be save the same as the names from the dictionary.

    Args:
        outputFileHDF5 (h5py.File type or str): The h5py created file or the path to the HDF5 file
        dic (dictionary): A python dictionary containing all parameters (metadata) from the experiment
        software (string): Name of the python module 'software'  used
        version (string): Version of python module used (can be called by: 'software.__version__'; example: 'sscRaft.__version__')

    """
    dic['Software'] = software
    dic['Version']  = version

    if isinstance(outputFileHDF5, str):

            if os.path.exists(outputFileHDF5):
                    hdf5 = h5py.File(outputFileHDF5, 'a')
            else:
                    hdf5 = h5py.File(outputFileHDF5, 'w')
    else:
            hdf5 = outputFileHDF5


    for key, value in dic.items():

            h5_path = 'Recon Parameters' #os.path.join('Recon Parameters', key)
            hdf5.require_group(h5_path)
            
            if key == 'findRotationAxis':
                    value = str(value)
            if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, numpy.ndarray):
                    value = numpy.asarray(value)
                    hdf5[h5_path].create_dataset(key, data=value, shape=value.shape)
            else:
                    # print(value)
                    hdf5[h5_path].create_dataset(key, data=value, shape=())

    if isinstance(outputFileHDF5, str):
            hdf5.close()

def power_of_2_padding(size,pad):
    return int((pow(2, numpy.ceil(numpy.log2(size + 2 * pad))) - size) * 0.5)
       
def set_conical_slices(slice_recon_start,slice_recon_end,nslices,nx,ny,z1,z12,pixel_det):
        
    magn       = z1/z12
    v          = nslices * pixel_det / 2
    dx, dy, dz = pixel_det*magn, pixel_det*magn, pixel_det*magn
    x, y, z    = dx*nx/2, dy*ny/2, dz*nslices/2
    L          = numpy.sqrt(x*x + y*y)

    z_min_recon = - z + slice_recon_start * dz
    z_max_recon = - z + slice_recon_end   * dz

    Z_min_proj = max(-v, min(z12*z_min_recon/(z1 - L), z12*z_min_recon/(z1 + L)))
    Z_max_proj = min( v, max(z12*z_max_recon/(z1 + L), z12*z_max_recon/(z1 - L)))

    slice_projection_start = max(0      , int(numpy.floor((Z_min_proj + v)/pixel_det)))
    slice_projection_end   = min(nslices, int(numpy.ceil( (Z_max_proj + v)/pixel_det)))

    return slice_projection_start, slice_projection_end

def set_conical_tomogram_slices(tomogram, dic):

    nrays   = tomogram.shape[-1]
    nangles = tomogram.shape[-2]    
    nslices = tomogram.shape[ 0]

    z1, z12   = dic['z1[m]'], dic['z1+z2[m]']
    pixel_det = dic['detectorPixel[m]']
    
    start_recon_slice = dic['slices'][0]
    end_recon_slice   = dic['slices'][1] 
    
    blockslices = end_recon_slice - start_recon_slice

    try:
        reconsize = dic['reconSize']

        if isinstance(reconsize,list) or isinstance(reconsize,tuple):
        
            if len(dic['reconSize']) == 1:
                nx, ny, nz = int(reconsize[0]), int(reconsize[0]), int(blockslices)
            else:
                nx, ny, nz = int(reconsize[0]), int(reconsize[1]), int(blockslices)

        elif isinstance(reconsize,int):
            nx, ny, nz = int(reconsize), int(reconsize), int(blockslices)
        else:
            logger.error(f'Dictionary entry `reconsize` wrong ({reconsize}). The entry `reconsize` is optional, but if it exists it needs to be a list = [nx,ny].')
            logger.error(f'Finishing run...')
            sys.exit(1)

    except:
        nx, ny, nz = int(nrays), int(nrays), int(blockslices)
        logger.info(f'Set default reconstruction size ({nz},{ny},{nx}) = (z,y,x).')

    
    start_tomo_slice,end_tomo_slice = set_conical_slices(start_recon_slice,end_recon_slice,nslices,nx,ny,z1,z12,pixel_det)

    _tomo_ = tomogram[start_tomo_slice:(end_tomo_slice + 1),:,:]

    dic.update({'slice tomo': [start_tomo_slice,end_tomo_slice]})

    return _tomo_, start_tomo_slice, end_tomo_slice

if __name__ == "__main__":
   pass

