# Authors: Giovanni L. Baraldi, Gilberto Martinez

from ...rafttypes import *
import numpy as np
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t

def fbpMultiGPU(tomogram, dic):
        
        gpus = dic['gpu']     
        ngpus = len(gpus)

        gpus = numpy.array(gpus)
        gpus = np.ascontiguousarray(gpus.astype(np.intc))
        gpusptr = gpus.ctypes.data_as(void_p)

        nrays   = tomogram.shape[-1]
        nangles = tomogram.shape[-2]
        
        if len(tomogram.shape) == 2:
                nslices = 1
        else:
                nslices = tomogram.shape[0]

        precision = dic['precision'].lower()
        if precision == 'float32':
                precision = 5
                recondtype = np.float32
        elif precision == 'uint16':
                precision = 2
                recondtype = np.uint16
        elif precision == 'uint8':
                precision = 1
                recondtype = np.uint8
        else:
                logger.error(f'Invalid recon datatype:{precision}')
        
        reconsize = dic['reconSize']

        angles = dic['angles']
        filter = int32(FilterNumber(dic['filter']))
        regularization = float32(dic['regularization'])
        threshold = float32(dic['threshold'])

        tomogram = np.ascontiguousarray(tomogram.astype(np.float32))
        tomogramptr = tomogram.ctypes.data_as(void_p)

        output = np.zeros((nslices,reconsize,reconsize),dtype=recondtype)
        outputptr = output.ctypes.data_as(void_p)

        try:
                angles    = np.ascontiguousarray(angles.astype(np.float32))
                anglesptr = angles.ctypes.data_as(void_p)
        except:
                anglesptr = void_p(0)

        tomooffset = dic['tomooffset']
        
        reconsize = int32(reconsize)
        tomooffset = int32(tomooffset)
        nrays = int32(nrays)
        nangles = int32(nangles)
        nslices = int32(nslices)
        bShiftCenter = int32(dic['shift center'])

        libraft.fbpblock(gpusptr, int32(ngpus), outputptr, tomogramptr, nrays, nangles, nslices, reconsize, tomooffset, regularization, anglesptr, threshold, precision, filter, bShiftCenter)

        return output

def fbpGPU(tomogram, dic, gpu = 0):
        
        ngpus   = gpu

        nrays   = tomogram.shape[-1]
        nangles = tomogram.shape[-2]
        
        if len(tomogram.shape) == 2:
                nslices = 1
        else:
                nslices = tomogram.shape[0]
        
        precision = dic['precision'].lower()

        if precision == 'float32':
                precision = 5
                recondtype = np.float32
        elif precision == 'uint16':
                precision = 2
                recondtype = np.uint16
        elif precision == 'uint8':
                precision = 1
                recondtype = np.uint8
        else:
                logger.error(f'Invalid recon datatype:{precision}')
        
        reconsize = dic['reconSize']

        angles = dic['angles']
        filter = int32(FilterNumber(dic['filter']))
        regularization = float32(dic['regularization'])
        threshold = float32(dic['threshold'])

        tomogram = np.ascontiguousarray(tomogram.astype(np.float32))
        tomogramptr = tomogram.ctypes.data_as(void_p)

        output = np.zeros((nslices,reconsize,reconsize),dtype=recondtype)
        outputptr = output.ctypes.data_as(void_p)

        try:
                angles = np.ascontiguousarray(angles.astype(np.float32))
                anglesptr = angles.ctypes.data_as(void_p)
        except:
                anglesptr = void_p(0)

        tomooffset = dic['tomooffset']
        
        reconsize = int32(reconsize)
        tomooffset = int32(tomooffset)
        nrays = int32(nrays)
        nangles = int32(nangles)
        nslices = int32(nslices)
        bShiftCenter = int32(dic['shift center'])

        libraft.fbpgpu(int32(ngpus), outputptr, tomogramptr, nrays, nangles, nslices, reconsize, tomooffset, regularization, anglesptr, threshold, precision, filter, bShiftCenter)

        return output



def fbp(tomogram, dic, **kwargs):
        
        nrays = tomogram.shape[-1]

        dicparams = ('gpu','angles','filter','reconSize','precision','regularization','threshold',
                    'shift center','tomooffset')
        defaut = ([0],None,'lorentz',nrays,'float32',1,0,False,0)
        
        SetDictionary(dic,dicparams,defaut)

        gpus  = dic['gpu']

        # reconsize = dic['reconSize']

        # if reconsize % 32 != 0:
        #         reconsize += 32-(reconsize%32)
        #         logger.info(f'Reconsize not multiple of 32. Setting to: {reconsize}')

        # dic.update({'reconSize': reconsize})

        if len(gpus) == 1:
                gpu = gpus[0]
                output = fbpGPU( tomogram, dic, gpu )
        else:
                output = fbpMultiGPU( tomogram, dic ) 

        # Garbage Collector
        # lists are cleared whenever a full collection or
        # collection of the highest generation (2) is run
        # collected = gc.collect() # or gc.collect(2)
        # logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')

        return output
