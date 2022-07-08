# Authors: Giovanni L. Baraldi, Gilberto Martinez

from ..rafttypes import *
import numpy as np
import gc
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t
import uuid
import SharedArray as sa


def flatdarkMultiGPU(frames, flat, dark, dic):
        
        gpus = dic['gpu']     
        ngpus = len(gpus)

        gpus = numpy.array(gpus)
        gpus = np.ascontiguousarray(gpus.astype(np.int32))
        gpusptr = gpus.ctypes.data_as(void_p)

        nrays   = frames.shape[-1]
        nangles = frames.shape[0]
        
        if len(frames.shape) == 2:
                nslices = 1
        else:
                nslices = frames.shape[-2]

        nflats = flat.shape[0]
        flat = np.ascontiguousarray(flat.astype(np.float32))
        flatptr = flat.ctypes.data_as(void_p)

        dark = np.ascontiguousarray(dark.astype(np.float32))
        darkptr = dark.ctypes.data_as(void_p)
        
        Is360pan = dic['360pan']

        frames = np.ascontiguousarray(frames.astype(np.float32))
        framesptr = frames.ctypes.data_as(void_p)

        print("valores python:",ngpus,gpus,nrays,nangles,nslices,nflats,frames.shape,flat.shape,dark.shape)
        
        nrays   = int32(nrays)
        nangles = int32(nangles)
        nslices = int32(nslices)
        nflats  = int32(nflats)
        
        libraft.flatdarktransposeblock(gpusptr, int32(ngpus), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats)

        return frames #np.swapaxes(frames,0,1)


def flatdarkGPU(frames, flat, dark, dic, gpu = 0):
        
        ngpus   = gpu

        nrays   = frames.shape[-1]
        nangles = frames.shape[0]
        
        if len(frames.shape) == 2:
                nslices = 1
        else:
                nslices = frames.shape[-2]

        nflats = flat.shape[0]
        flat = np.ascontiguousarray(flat.astype(np.float32))
        flatptr = flat.ctypes.data_as(void_p)

        dark = np.ascontiguousarray(dark.astype(np.float32))
        darkptr = dark.ctypes.data_as(void_p)
        
        Is360pan = dic['360pan']

        frames = np.ascontiguousarray(frames.astype(np.float32))
        framesptr = frames.ctypes.data_as(void_p)
        
        nrays   = int32(nrays)
        nangles = int32(nangles)
        nslices = int32(nslices)
        nflats  = int32(nflats)

        libraft.flatdarktransposegpu(int32(ngpus), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats)

        return frames #np.swapaxes(frames,0,1)

def _worker_flatdark_(params, start, end, gpu, process):

        output = params[0]
        data   = params[1]
        dic    = params[6]
        flat   = params[7]
        dark   = params[8]

        logger.info(f'flatdark: begin process {process} on gpu {gpu}')

        output[start:end,:,:] = flatdarkGPU( data[:,start:end,:], flat[:,start:end,:], dark[start:end,:], dic, gpu )

        logger.info(f'flatdark: end process {process} on gpu {gpu}')
    

def _build_flatdark_(params):
 
        nslices = params[2]
        gpus    = params[5]
        ngpus = len(gpus)

        b = int( numpy.ceil( nslices/ngpus )  ) 

        processes = []
        for process in range( ngpus ):
                begin_ = process*b
                end_   = min( (process+1)*b, nslices )

                p = multiprocessing.Process(target=_worker_flatdark_, args=(params, begin_, end_, gpus[process], process))
                processes.append(p)
    
        for p in processes:
                p.start()

        for p in processes:
                p.join()


def flatdark_gpublock( frames, flat, dark, dic ):

        nslices     = frames.shape[1]
        nangles     = frames.shape[0]
        nrays       = frames.shape[2]
        gpus        = dic['gpu']

        name = str( uuid.uuid4())

        try:
                sa.delete(name)
        except:
                pass

        output  = sa.create(name,[nslices, nangles, nrays], dtype=np.float32)

        _params_ = ( output, frames, nslices, nangles, nrays, gpus, dic, flat, dark)

        _build_flatdark_( _params_ )

        sa.delete(name)

        return output


def flatdarkLog_threads(frames, flat, dark, dic, **kwargs):
        
        dicparams = ('gpu','360pan')
        defaut = ([0],False)
        
        SetDictionary(dic,dicparams,defaut)

        gpus  = dic['gpu']

        if len(gpus) == 1:
                gpu = gpus[0]
                output = flatdarkGPU( frames, flat, dark, dic, gpu )
        else:
                output = flatdark_gpublock( frames, flat, dark, dic ) 

        return np.swapaxes(output,1,0)


def flatdarkLog(frames, flat, dark, dic, **kwargs):
        
        dicparams = ('gpu','360pan')
        defaut = ([0],False)
        
        SetDictionary(dic,dicparams,defaut)

        gpus  = dic['gpu']

        if len(gpus) == 1:
                gpu = gpus[0]
                output = flatdarkGPU( frames, flat, dark, dic, gpu )
        else:
                output = flatdarkMultiGPU( frames, flat, dark, dic ) 

        return np.swapaxes(output,1,0)