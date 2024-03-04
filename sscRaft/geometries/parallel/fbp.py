# Authors: Giovanni L. Baraldi, Gilberto Martinez

from ...rafttypes import *
from ...processing.io import *
import numpy as np
from time import time

def fbpGPU(tomogram, angles, gpus, dic):
           
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = np.ascontiguousarray(gpus.astype(np.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = tomogram.shape[-1]
    nangles  = tomogram.shape[-2]
    
    if len(tomogram.shape) == 2:
            nslices = 1
    else:
            nslices = tomogram.shape[0]
    
    objsize        = nrays

    filter_type    = FilterNumber(dic['filter'])
    paganin        = dic['paganin regularization']
    regularization = dic['regularization']
    offset         = dic['offset']

    # logger.info(f'FBP Paganin regularization: {paganin}')

    padx, pady, padz  = dic['padding'],0,0 # (padx, pady, padz)

    # pad = padx * nrays
    # logger.info(f'Set FBP pad value as {padx} x horizontal dimension = ({pad}).')

    tomogram     = CNICE(tomogram) 
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    obj          = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
    obj_ptr      = obj.ctypes.data_as(ctypes.c_void_p)

    angles       = numpy.array(angles)
    angles       = CNICE(angles) 
    angles_ptr   = angles.ctypes.data_as(ctypes.c_void_p) 

    param_int     = [nrays, nangles, nslices, objsize, 
                     padx, pady, padz, filter_type, offset]
    param_int     = numpy.array(param_int)
    param_int     = CNICE(param_int,numpy.int32)
    param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

    param_float     = [paganin, regularization]
    param_float     = numpy.array(param_float)
    param_float     = CNICE(param_float,numpy.float32)
    param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)


    # bShiftCenter = dic['shift center']

    libraft.getFBPMultiGPU(gpus_ptr, ctypes.c_int(ngpus), 
        obj_ptr, tomogram_ptr, angles_ptr, 
        param_float_ptr, param_int_ptr)

    return obj

def fbp(tomogram, dic, angles = None, **kwargs):
    """Computes the Reconstruction of a Parallel Tomogram using the 
    Filtered Backprojection method (FBP).
    GPU function.

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        dic (dict): Dictionary with the experiment info.

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

    * One or MultiGPUs. 
    * Calls function ``fbpGPU()``.

    Dictionary parameters:

        * ``dic['gpu']`` (ndarray): List of gpus for processing [required]
        * ``dic['angles[rad]']`` (list): list of angles in radians [required]
        * ``dic['filter']`` (str,optional): Filter type [default: \'lorentz\']

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')
            
        * ``dic['paganin regularization']`` (float,optional): Paganin regularization value ( value >= 0 ) [default: 0.0]
        * ``dic['regularization']`` (float,optional): Regularization value ( value >= 0 ) [default: 1.0]  
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 2]  

    """      
    required = ('gpu',)
    optional = ( 'filter','offset','padding','regularization','paganin regularization')
    default  = ('lorentz',       0,        2,             1.0,                     0.0)
    
    SetDictionary(dic,required,optional,default)

    gpus  = dic['gpu']

    # objsize = dic['objsize']
    # if objsize % 32 != 0:
    #         objsize += 32-(objsize%32)
    #         logger.info(f'Reconstruction size not multiple of 32. Setting to: {objsize}')
    # dic.update({'objsize': objsize})

    try:
        angles = dic['angles[rad]']
    except:
        if angles is None:
            logger.error(f'Missing angles list!! Finishing run...') 
            raise ValueError(f'Missing angles list!!')

    output = fbpGPU( tomogram, angles, gpus, dic ) 

    return output
