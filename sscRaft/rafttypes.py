#We use the one encoding: utf8
import ctypes
from ctypes import *
import ctypes.util
import multiprocessing
import os
import sys
import numpy
import gc
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

if sys.version_info[0] >= 3:
    import sysconfig
    ext = sysconfig.get_config_var('SO')
else:
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

######## EM ##########

try:
    libraft.tEM.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           c_int, c_int, c_int, c_int, c_int, c_int]
    libraft.tEM.restype  = None

    libraft.eEM.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            c_int, c_int, c_int, c_int, c_int, c_int]
    libraft.eEM.restype  = None

    libraft.EMTV.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
                             c_float, c_float]
    libraft.EMTV.restype  = None

    # EM threads in C
    libraft.tEMblock.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           c_int, c_int, c_int, c_int, c_int, c_int, ctypes.c_void_p]
    libraft.tEMblock.restype  = None

    libraft.eEMblock.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            c_int, c_int, c_int, c_int, c_int, c_int, ctypes.c_void_p]
    libraft.eEMblock.restype  = None

    libraft.tEMgpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           c_int, c_int, c_int, c_int, c_int, c_int]
    libraft.tEMgpu.restype  = None

    libraft.eEMgpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            c_int, c_int, c_int, c_int, c_int, c_int]
    libraft.eEMgpu.restype  = None

    libraft.emfreq.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            c_int, c_int, c_int, c_int, c_int, c_float, c_int, c_int]
    libraft.emfreq.restype  = None

    libraft.emfreqblock.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            c_int, c_int, c_int, c_int, c_int, c_float, c_int, c_int, ctypes.c_void_p]
    libraft.emfreqblock.restype  = None

except:
    print('-.RAFT_PARALLEL_EM-')
    pass

######## Rebinning ##########

try:
    libraft.CPUrebinning.argtypes = [ ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    libraft.CPUrebinning.restype  = None
    
    libraft.GPUrebinning.argtypes = [ ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    libraft.GPUrebinning.restype  = None
    
except:
    print('-.REB-')
    pass

######## Parallel Raft ##########

try:
    libraft.ComputeTomo360Offsetgpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.ComputeTomo360Offsetgpu.restype = ctypes.c_int

    libraft.ComputeTomo360Offset16.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.ComputeTomo360Offset16.restype = ctypes.c_int

    libraft.Tomo360To180gpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.Tomo360To180gpu.restype  = None

    libraft.Tomo360To180block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.Tomo360To180block.restype  = None

    libraft.findcentersino.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                        ctypes.c_int, ctypes.c_int]
    libraft.findcentersino.restype = ctypes.c_int

    libraft.findcentersino16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                        ctypes.c_int, ctypes.c_int]
    libraft.findcentersino16.restype = ctypes.c_int

    libraft.ringsgpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_float, ctypes.c_int]
    libraft.ringsgpu.restype  = None

    libraft.ringsblock.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_float, ctypes.c_int]
    libraft.ringsblock.restype  = None

    libraft.fbpgpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_float, 
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.fbpgpu.restype  = None

    libraft.fbpblock.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_float, 
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.fbpblock.restype  = None
   
    libraft.flatdark_gpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.flatdark_gpu.restype  = None
        
    libraft.flatdark_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.flatdark_block.restype  = None

    
except:
    print('-.RAFT_PARALLEL-')
    pass

######## Paganin ##########

try:
    libraft.phase_filters.argtypes = [  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_void_p, ctypes.c_int]
    libraft.phase_filters.restype  = None
    
except:
    print('-.PHASE_FILTERS-')
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
                ("slice0", ctypes.c_int),
                ("slice1", ctypes.c_int),
                ("nslices", ctypes.c_int)
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
                                        print('Using default -', paramname,':', deff[i])
                                        dic[paramname][i] = deff[i]

        except:
                print('Using default -', paramname,':', deff)
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
                
                if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, numpy.ndarray):
                       value = numpy.asarray(value)
                       hdf5[h5_path].create_dataset(key, data=value, shape=value.shape)
                else:
                       if key == 'recon type':
                                value = str(value)
                       
                       hdf5[h5_path].create_dataset(key, data=value, shape=())

        if isinstance(outputFileHDF5, str):
               hdf5.close()

def power_of_2_padding(size,pad):

    n = numpy.log2(size + 2 * pad)
    r = n % 1
    if r != 0:
        power = n + 1 - r
        pad  = int(pow(2,power) - size ) // 2

    return pad

if __name__ == "__main__":
   pass

