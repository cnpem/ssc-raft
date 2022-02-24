#We use the one encoding: utf8
import ctypes
from ctypes import *
import ctypes.util
import multiprocessing
import math
import os
import sys
import numpy
import time

nthreads = multiprocessing.cpu_count()

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
    try:
        lib = ctypes.CDLL(_path)
        return lib
    except:
        pass
    return None


libraft  = load_library(_lib, ext)

#########################

try:
    libraft.EM.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           c_int, c_int, c_int, c_int, c_int, c_int]
    libraft.EM.restype  = None

except:
    pass


##############
#|          |#
##############

if __name__ == "__main__":
   pass

