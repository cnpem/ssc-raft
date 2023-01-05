#We use the one encoding: utf8
import ctypes
from ctypes import *
import ctypes.util
import multiprocessing
import os
import sys
import numpy
import logging
import json

'''----------------------------------------------'''

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

'''----------------------------------------------'''

nthreads = multiprocessing.cpu_count()
# try:
#    multiprocessing.set_start_method('spawn', force=True)
#    print("spawned")
# except RuntimeError:
#    pass


# Load required libraies:

libstdcpp = ctypes.CDLL( ctypes.util.find_library( "stdc++" ), mode=ctypes.RTLD_GLOBAL )
libblas   = ctypes.CDLL( ctypes.util.find_library( "blas" ), mode=ctypes.RTLD_GLOBAL )
libfftw3  = ctypes.CDLL( ctypes.util.find_library( "fftw3" ), mode=ctypes.RTLD_GLOBAL )
libfftw3_threads  = ctypes.CDLL( ctypes.util.find_library( "fftw3_threads" ), mode=ctypes.RTLD_GLOBAL )


_lib = "lib/libraft"

if sys.version_info[0] >= 3:
    import sysconfig
    ext = sysconfig.get_config_var('SO')
else:
    ext = '.so'

def load_library(lib,ext):
    _path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib + ext
#     print(_path)
#     try:
    lib = ctypes.CDLL(_path)
    print(lib)
    return lib
#     except:
#         pass
#     return None

libraft  = load_library(_lib, ext)

#########################

#########################
#|       ssc-raft      |#
#| Function prototypes |#
#########################

######## EM ##########

try:
    libraft.tEM.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           c_int, c_int, c_int, c_int, c_int, c_int]
    libraft.tEM.restype  = None

    libraft.eEM.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                            c_int, c_int, c_int, c_int, c_int, c_int]
    libraft.eEM.restype  = None

    libraft.EMTV.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                             c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
                             c_float, c_float]
    libraft.EMTV.restype  = None

except:
    print('-.EM-')
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

    libraft.bstgpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    libraft.bstgpu.restype  = None

    libraft.bstblock.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    libraft.bstblock.restype  = None

    libraft.fstgpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    libraft.fstgpu.restype  = None
    
    libraft.fstblock.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    libraft.fstblock.restype  = None   
   
    libraft.flatdarktransposegpu.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.flatdarktransposegpu.restype  = None
        
    libraft.flatdarktransposeblock.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libraft.flatdarktransposeblock.restype  = None
    
except:
    print('-.RAFT-')
    pass



######## Conical Raft ##########

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
                ("nbeta", ctypes.c_int)]

try:
        libraft.gpu_fdk.argtypes = [Lab, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p ]


except:
    print('-.FDK-')
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
        if mfilter.lower() == 'gaussian' or mfilter.lower() == 'gauss':
                return 1
        elif mfilter.lower() == 'lorentz':
                return 2
        elif mfilter.lower() == 'cosine' or mfilter.lower() == 'cos':
                return 3
        elif mfilter.lower() == 'rectangle' or mfilter.lower() == 'rect':
                return 4
        elif mfilter.lower() == 'hann':
                return 5
        elif mfilter.lower() == 'FSC':
                return 0
        
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
        except:
                print('Using default -', paramname,':', deff)
                dic[paramname] = deff


def SetDictionary(dic,param,default):
        for ind in range(len(param)):
                SetDic(dic,param[ind], default[ind])


if __name__ == "__main__":
   pass

