from ..rafttypes import *
import numpy

def log(data):
    """CPU float function to apply logarithm on an array.

    Args:
        tomogram (ndarray): data. Axes are [z,y,x].

    Returns:
        (ndarray): data. Axes are [z,y,x].
    """
    dim = data.shape
    
    total_size = 1
    for i in range(dim):
        total_size *= dim[i]
    
    data     = CNICE(data)
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)

    libraft.logf_cpu( data_ptr, ctypes.c_int(total_size) ) 

    return data

def exp(data):
    """CPU float function to apply exponential on an array.

    Args:
        tomogram (ndarray): data. Axes are [z,y,x].

    Returns:
        (ndarray): data. Axes are [z,y,x].
    """
    dim = data.shape
    
    total_size = 1

    for i in range(dim):
        total_size *= dim[i]
    
    data     = CNICE(data)
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)

    libraft.expf_cpu( data_ptr, ctypes.c_int(total_size) ) 

    return data

def transpose(data):
    """CPU float function to apply transpose on a 3D array.

    Args:
        tomogram (ndarray): data. Axes are [z,y,x].

    Returns:
        (ndarray): data. Axes are [z,y,x].
    """
    sizex = data.shape[-1]
    sizey = data.shape[-2]

    if len(data.shape) == 2:
        sizez = 1
    else:
        sizez = data.shape[0]

    data     = CNICE(data)
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)

    libraft.transpose_cpu( data_ptr, 
            ctypes.c_int(sizex), 
            ctypes.c_int(sizey), 
            ctypes.c_int(sizez) ) 

    return data