# Authors: Giovanni L. Baraldi, Gilberto Martinez

from ....rafttypes import *
import numpy as np
import gc
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t
import uuid
import SharedArray as sa


def fstMultiGPU(tomogram, dic):
        
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
        
        reconsize = dic['recon size']

        Is360pan = dic['360pan']
        angles = dic['angles']
        precision = int(dic['precision'])
        filter = int32(FilterNumber(dic['filter']))
        regularization = float32(dic['regularization'])
        recondtype = dic['recon type']
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

        if Is360pan:
                tomooffset = 0
        else:
                tomooffset = dic['tomooffset']
        
        reconsize = int32(reconsize)
        tomooffset = int32(tomooffset)
        nrays = int32(nrays)
        nangles = int32(nangles)
        nslices = int32(nslices)

        libraft.fstblock(gpusptr, int32(ngpus), outputptr, tomogramptr, nrays, nangles, nslices, reconsize, tomooffset, regularization, filter, anglesptr)

        return output


def fstGPU(tomogram, dic, gpu = 0):
        
        ngpus   = gpu

        nrays   = tomogram.shape[-1]
        nangles = tomogram.shape[-2]
        
        if len(tomogram.shape) == 2:
                nslices = 1
        else:
                nslices = tomogram.shape[0]
        
        reconsize = dic['recon size']

        Is360pan = dic['360pan']
        angles = dic['angles']
        precision = int(dic['precision'])
        filter = int32(FilterNumber(dic['filter']))
        regularization = float32(dic['regularization'])
        recondtype = dic['recon type']
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

        if Is360pan:
                tomooffset = 0
        else:
                tomooffset = dic['tomooffset']
        
        reconsize = int32(reconsize)
        tomooffset = int32(tomooffset)
        nrays = int32(nrays)
        nangles = int32(nangles)
        nslices = int32(nslices)

        libraft.fstgpu(int32(ngpus), outputptr, tomogramptr, nrays, nangles, nslices, reconsize, tomooffset, regularization, filter, anglesptr)

        return output

def _worker_fst_(params, start, end, gpu, process):

        output = params[0]
        data   = params[1]
        dic    = params[6]

        logger.info(f'FST: begin process {process} on gpu {gpu}')

        output[start:end,:,:] = fstGPU( data[start:end, :, :], dic, gpu )

        logger.info(f'FST: end process {process} on gpu {gpu}')
    

def _build_fst_(params):
 
        nslices = params[2]
        gpus    = params[5]
        ngpus = len(gpus)

        b = int( numpy.ceil( nslices/ngpus )  ) 

        processes = []
        for process in range( ngpus ):
                begin_ = process*b
                end_   = min( (process+1)*b, nslices )

                p = multiprocessing.Process(target=_worker_fst_, args=(params, begin_, end_, gpus[process], process))
                processes.append(p)
    
        for p in processes:
                p.start()

        for p in processes:
                p.join()


def fst_gpublock( tomogram, dic ):

        nslices     = tomogram.shape[0]
        nangles     = tomogram.shape[1]
        nrays       = tomogram.shape[2]
        gpus        = dic['gpu']
        outtype     = dic['recon type']

        name = str( uuid.uuid4())

        try:
                sa.delete(name)
        except:
                pass

        output  = sa.create(name,[nslices, nangles, nrays], dtype=outtype)

        _params_ = ( output, tomogram, nslices, nangles, nrays, gpus, dic)

        _build_fst_( _params_ )

        sa.delete(name)

        return output


def fst_threads(tomogram, dic, **kwargs):
        
        nrays = tomogram.shape[2]

        dicparams = ('gpu','angles','filter','recon size','precision','regularization','threshold',
                    'shift center','tomooffset','360pan')
        defaut = ([0],None,None,nrays,'float32',0,0,False,0,False)
        
        SetDictionary(dic,dicparams,defaut)

        gpus  = dic['gpu']

        reconsize = dic['recon size']

        if reconsize % 32 != 0:
                reconsize += 32-(reconsize%32)
                logger.info(f'Reconsize not multiple of 32. Setting to: {reconsize}')

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
        
        dic.update({'recon size': reconsize,'recon type': recondtype, 'precision': precision})

        if len(gpus) == 1:
                gpu = gpus[0]
                output = fstGPU( tomogram, dic, gpu )
        else:
                output = fst_gpublock( tomogram, dic ) 

        return output


def fst(tomogram, dic, **kwargs):
        
        nrays = tomogram.shape[-1]

        dicparams = ('gpu','angles','filter','recon size','precision','regularization','threshold',
                    'shift center','tomooffset','360pan')
        defaut = ([0],None,None,nrays,'float32',0,0,False,0,False)
        
        SetDictionary(dic,dicparams,defaut)

        gpus  = dic['gpu']

        reconsize = dic['recon size']

        if reconsize % 32 != 0:
                reconsize += 32-(reconsize%32)
                logger.info(f'Reconsize not multiple of 32. Setting to: {reconsize}')

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
        
        dic.update({'recon size': reconsize,'recon type': recondtype, 'precision': precision})

        if len(gpus) == 1:
                gpu = gpus[0]
                output = fstGPU( tomogram, dic, gpu )
        else:
                output = fstMultiGPU( tomogram, dic ) 

        return output