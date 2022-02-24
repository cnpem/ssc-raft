from ..rafttypes import *

import os
import ctypes

from ctypes import c_int as int32
from ctypes import c_float as float32
from ctypes import POINTER
from ctypes import c_void_p  as void_p

import cupy
import numpy
import sys
import gc
import uuid
import SharedArray as sa

import warnings

from sscRadon import radon
from sscBst import backprojection

def CNICE(darray,dtype=numpy.float32):
    if darray.dtype != dtype:
        return numpy.ascontiguousarray(darray.astype(dtype))
    elif darray.flags['C_CONTIGUOUS'] == False:
        return numpy.ascontiguousarray(darray)
    else:
        return darray


#LEGEND:
#------
#mpfsupy: (m)ulti(p)rocessing (f)rom (s)inograms (u)sing (py)thon
#mpfsucu: (m)ulti(p)rocessing (f)rom (s)inograms (u)sing (cu)da
#
#

def _iterations_em_mpfsupy_(sino, niter, device):
    
    counts = numpy.exp(-sino)
    cflat  = numpy.ones(sino.shape)
    block   = sino.shape[0]
    nangles = sino.shape[1]
    nrays   = sino.shape[2]
    recsize = sino.shape[2]
    tol     = 1e-4
    
    start = time.time()
    
    '''
    backcounts = backprojection.ss( counts, device)
    recon      = numpy.ones([block, recsize, recsize])
    
    for m in range(niter):
        _s_ = radon.radon_gpu( recon, nangles, device)
        _r_ = cflat * numpy.exp(-_s_)
        recon = recon * backprojection.ss( _r_, device) / backcounts
                
    return recon
    '''

    recon   = numpy.ones([block, recsize, recsize])
    recon_p = recon.ctypes.data_as(void_p)

    c   = CNICE(counts) #count pointer
    c_p = c.ctypes.data_as(void_p) 

    f   = CNICE(cflat) #flat pointer
    f_p = f.ctypes.data_as(void_p) 
    
    libraft.EM( recon_p, c_p, f_p, int32(recsize), int32(nrays), int32(nangles), int32(block),
                int32(device), int32(niter))

    elapsed = time.time() - start

    return recon

def _worker_em_mpfsupy_(params, idx_start,idx_end, gpu, blocksize):

    nblocks = ( idx_end + 1 - idx_start ) // blocksize

    output  = params[0]
    data    = params[1]
    niter   = params[6]
    
    for k in range(nblocks):
        _start_ = idx_start + k * blocksize
        _end_   = _start_ + blocksize
        
        print('--> ids,ide: GPU({})'.format(gpu), idx_start, idx_end )
        print('  > GPU[{}]'.format(gpu),_start_, _end_)
        output[_start_:_end_,:,:] = _iterations_em_mpfsupy_(  data[_start_:_end_, :, :],  niter, gpu )
        
    
def _build_em_mpfsupy_(params):

    #_params_ = ( output, data, nslices, nangles, gpus, blocksize)
 
    nslices = params[2]
    gpus    = params[4]
    blocksize = params[5]
    ngpus = len(gpus)
    
    b = int( numpy.ceil( nslices/ngpus )  ) 
    
    processes = []
    for k in range( ngpus ):
        begin_ = k*b
        end_   = min( (k+1)*b, nslices )

        p = multiprocessing.Process(target=_worker_em_mpfsupy_, args=(params, begin_, end_, gpus[k], blocksize ))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()

###########

def emfs( tomogram, dic ):

    """ Transmission expectation maximization for 3D tomographic parallel sinograms

    Args:
        tomogram: three-dimensional stack of sinograms 
        dic: input dictionary 
        
    Returns:
        (ndarray): stacking 3D reconstructed volume  

    * CPU function
    * This function uses a shared array through package 'SharedArray'.
    * The total number of images will be divided by the number of processes given as input
    * SharedArray names are provided by uuid.uuid4() 
    
    Parameters:
    
    * ``dic['nangles']``:  Number of angles
    * ``dic['gpu']``:  List of GPU devices used for computation 
    * ``dic['blocksize']``:  Number of images to compute parallel radon transform 
    * ``dic['niterations']``:  Number of iterations for the EM algorithm.
         
    """
    
    nslices   = tomogram.shape[0]
    nrays     = tomogram.shape[2]
    nangles   = dic['nangles']
    gpus      = dic['gpu']
    blocksize = dic['blocksize']
    niter     = dic['niterations']

    if blocksize > nslices // len(gpus):
        print('ssc-radon: Error! Please check block size!')
    
    name = str( uuid.uuid4())
    
    try:
        sa.delete(name)
    except:
        pass
        
    output  = sa.create(name,[nslices, nrays, nrays], dtype=numpy.float32)
    
    _params_ = ( output, tomogram, nslices, nangles, gpus, blocksize, niter)
    
    _build_em_mpfsupy_( _params_ )

    sa.delete(name)
    
    return output
