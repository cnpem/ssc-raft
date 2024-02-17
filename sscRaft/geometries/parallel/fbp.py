# Authors: Giovanni L. Baraldi, Gilberto Martinez

from ...rafttypes import *
import numpy as np
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t

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
        if filter_type == 0:
            dic['regularization'] = - 1.0 

        paganin        = dic['paganin regularization']
        regularization = dic['regularization']
        offset         = dic['offset']

        # logger.info(f'FBP Paganin regularization: {paganin}')

        padx           = int(nrays // 2)
        pad            = dic['padding']
        padx           = int(pad * padx)

        logger.info(f'Set FBP pad value as {pad} x horizontal dimension ({padx}).')

        tomogram     = CNICE(tomogram) 
        tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

        obj          = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
        obj_ptr      = obj.ctypes.data_as(ctypes.c_void_p)

        angles       = numpy.array(angles)
        angles       = CNICE(angles) 
        angles_ptr   = angles.ctypes.data_as(ctypes.c_void_p) 

        param_int     = [nrays, nangles, nslices, objsize, padx, padx, filter_type, offset]
        param_int     = numpy.array(param_int)
        param_int     = CNICE(param_int,numpy.int32)
        param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

        param_float     = [regularization, paganin]
        param_float     = numpy.array(param_float)
        param_float     = CNICE(param_float,numpy.float32)
        param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)

        # bShiftCenter = dic['shift center']

        libraft.getFBPMultiGPU(
                gpus_ptr, ctypes.c_int(ngpus), 
                obj_ptr, tomogram_ptr, angles_ptr, 
                param_float_ptr, param_int_ptr)

        return obj


def fbp(tomogram, dic, angles = None, **kwargs):
        
        dicparams = ( 'filter','offset','padding','regularization', 'paganin regularization')
        defaut    = ('lorentz',       0,        2,             1.0,                      0.0)
        
        SetDictionary(dic,dicparams,defaut)

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

        output = fbpPU( tomogram, angles, gpus, dic ) 

        return output
