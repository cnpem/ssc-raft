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

#from sscRadon import radon
#from sscBst import backprojection

def CNICE(darray,dtype=numpy.float32):
    if darray.dtype != dtype:
        return numpy.ascontiguousarray(darray.astype(dtype))
    elif darray.flags['C_CONTIGUOUS'] == False:
        return numpy.ascontiguousarray(darray)
    else:
        return darray


#LEGEND:
#------
# mpfs: (m)ulti(p)rocessing (f)rom (s)inograms 
#

def _iterations_emtv_mpfs_(sino, niter, device, reg, eps, process, angles):

    '''
    #Python version, for completeness: depends on ssc-radon/ssc-bst

    block   = sino.shape[0]
    nangles = sino.shape[1]
    nrays   = sino.shape[2]
    recsize = sino.shape[2]

    def iterem(x, sino, nangles, device, b):
        rad = radon.radon_gpu(x, nangles, device)
        y = sino/(rad) 
        y[ numpy.isnan(y)] = 0
        u = x * backprojection.ss( y, device) / b
        error = numpy.linalg.norm( u - x)
        return u, error, rad

    def itertv(y, x, sino, nangles, device, b, reg, eps):
        sqrtA = eps + (numpy.roll(y,1,0) - y)**2 + (numpy.roll(y,1,1) - y)**2
        A     = -reg * b * numpy.sqrt( sqrtA )
        
        sqrtB = eps + (y - numpy.roll(y,-1,0))**2 + (numpy.roll(numpy.roll(y,-1,0),1,1) - numpy.roll(y,-1,0))**2 
        B     =  reg * b * numpy.sqrt( sqrtB )
        C     = A
        
        sqrtD = eps + (numpy.roll(numpy.roll(y,1,0),-1,1) - numpy.roll(y,-1,1))**2 + (y - numpy.roll(y,-1,1))**2
        D     = reg * b * numpy.sqrt( sqrtD )
        
        rhs = x - y * ( numpy.roll(y,1,0)/A - numpy.roll(y,-1,0)/B + numpy.roll(y,1,1)/C - numpy.roll(y,-1,1)/D )
        u   = rhs / ( y * (-1/A + 1/B - 1/C + 1/D) + 1)
        
        error = numpy.linalg.norm(u - y)                    
        return u, error
    
    sinoones = numpy.ones(sino.shape)
    backones = backprojection.ss( sinoones, device)
    
    x = numpy.ones([block, recsize, recsize])
    x, error, _ = iterem(x, sino, nangles, device, backones)

    niter_em = niter[1]
    niter_tv = niter[2]
  
    for m in range(niter[0]):

        if True:
            tmp = numpy.copy(x)
            
            #EM step
            for k in range(niter_em):
                x,_, rad = iterem(x, sino, nangles, device, backones)
                
            #TV step
            y = numpy.copy(x) 

            for k in range(niter_tv):
                y,_ = itertv(y, x, sino, nangles, device, backones, reg, eps) 

            x = numpy.copy(y)

            _error_ = numpy.linalg.norm(x - tmp)
            
            print('EMTV/Process[{}]/Iter[{}]:'.format(process,m),_error_, error)
            if _error_ > error:
                break
            else:
                error = _error_
            ###
            
    return x
    '''

    #CUDA version
    block   = sino.shape[0]
    nangles = sino.shape[1]
    nrays   = sino.shape[2]
    recsize = sino.shape[2]
    
    start = time.time()

    x   = numpy.zeros([block, recsize, recsize], dtype=numpy.float32)
    x_p = x.ctypes.data_as(void_p)

    s   = CNICE(sino) #sino pointer
    s_p = s.ctypes.data_as(void_p) 

    a   = CNICE(angles) #angles pointer
    a_p = a.ctypes.data_as(void_p) 

    libraft.EMTV( x_p, s_p, a_p,
                  int32(recsize), int32(nrays),
                  int32(nangles), int32(block),
                  int32(device),  int32(niter[0]),
                  int32(niter[1]),int32(niter[2]),
                  float32(reg), float32(eps))
    
    elapsed = time.time() - start

    return x

def _iterations_eem_mpfs_(sino, niter, device, reg, eps, process, angles):

    '''
    #Python version, for completeness: depends on ssc-radon/ssc-bst

    block   = sino.shape[0]
    nangles = sino.shape[1]
    nrays   = sino.shape[2]
    recsize = sino.shape[2]
        
    sinoones = numpy.ones(sino.shape)
    backones = backprojection.ss( sinoones, device)
    
    def iterem(x, sino, nangles, device, b):
        rad = radon.radon_gpu(x,  nangles, device )
        y = sino/(rad)
        y[ rad==0 ] = 0
        u = x * backprojection.ss( y, device) / b
        error = numpy.linalg.norm( u - x)
        return u, error, rad

    x = numpy.ones([block, recsize, recsize])
    x, error, _ = iterem(x, sino, nangles, device, backones)

    for m in range(niter[0]):
        x, error2, rad = iterem(x, sino, nangles, device, backones)
        if error2 > error:
            break
        else:
            error = error2
            
    return x
    '''

    #CUDA version
    block   = sino.shape[0]
    nangles = sino.shape[1]
    nrays   = sino.shape[2]
    recsize = sino.shape[2]
    
    start = time.time()

    x   = numpy.zeros([block, recsize, recsize], dtype=numpy.float32)
    x_p = x.ctypes.data_as(void_p)

    s   = CNICE(sino) #sino pointer
    s_p = s.ctypes.data_as(void_p) 

    a   = CNICE(angles) #angles pointer
    a_p = a.ctypes.data_as(void_p) 
    
    libraft.eEM( x_p, s_p, a_p,
                 int32(recsize), int32(nrays),
                 int32(nangles), int32(block),
                 int32(device),  int32(niter[0]))

    elapsed = time.time() - start

    return x


def _iterations_tem_mpfs_(sino, niter, device, reg, eps, process, angles):

    '''
    #Python version, for completeness: depends on ssc-radon/ssc-bst

    counts = numpy.exp(-sino)
    cflat  = numpy.ones(sino.shape)
    
    block   = sino.shape[0]
    nangles = sino.shape[1]
    nrays   = sino.shape[2]
    recsize = sino.shape[2]
    
    backcounts = backprojection.ss( counts, device)
    x          = numpy.ones([block, recsize, recsize])
    
    for m in range(niter[0]):
        _s_ = radon.radon_gpu( x, nangles, device)
        _r_ = cflat * numpy.exp(-_s_)
        x = x * backprojection.ss( _r_, device) / backcounts
                
    return x
    '''

    #CUDA version
    counts = numpy.exp(-sino)
    cflat  = numpy.ones(sino.shape)
    block   = sino.shape[0]
    nangles = sino.shape[1]
    nrays   = sino.shape[2]
    recsize = sino.shape[2]
    
    start = time.time()

    x   = numpy.zeros([block, recsize, recsize], dtype=numpy.float32)
    x_p = x.ctypes.data_as(void_p)

    c   = CNICE(counts) #count pointer
    c_p = c.ctypes.data_as(void_p) 

    f   = CNICE(cflat) #flat pointer
    f_p = f.ctypes.data_as(void_p) 

    a   = CNICE(angles) #angles pointer
    a_p = a.ctypes.data_as(void_p) 
    
    libraft.tEM( x_p, c_p, f_p, a_p, 
                 int32(recsize), int32(nrays),
                 int32(nangles), int32(block),
                 int32(device), int32(niter[0]))
    
    elapsed = time.time() - start

    return x
    
def _worker_em_mpfs_(params, idx_start,idx_end, gpu, blocksize,process):

    nblocks = ( idx_end + 1 - idx_start ) // blocksize + 1

    output1 = params[0]
    data    = params[1]
    niter   = params[6]
    reg     = params[7]
    eps     = params[8]
    method  = params[9]
    angles  = params[10]

    if method=="tEM":
        InversionMethod = _iterations_tem_mpfs_
    elif method=="eEM":
        InversionMethod = _iterations_eem_mpfs_
    elif method=="EMTV":
        InversionMethod = _iterations_emtv_mpfs_
    else:
        print('ssc-raft: Error! Wrong method: "tEM"/"eEM"/"EMTV".')
        return
    
    
    for k in range(nblocks):
        _start_ = idx_start + k * blocksize
        _end_   = min( _start_ + blocksize, idx_end) 
        #print('--> Process {}: GPU({}) / [{},{}]'.format(process, gpu, _start_, _end_) )
        output1[_start_:_end_,:,:] = InversionMethod( data[_start_:_end_, :, :], niter, gpu, reg, eps, process, angles)
        
        
        
def _build_em_mpfs_(params):

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

        p = multiprocessing.Process(target=_worker_em_mpfs_, args=(params, begin_, end_, gpus[k], blocksize,k))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()

###########

def emfs( tomogram, dic ):

    """ Expectation maximization for 3D tomographic parallel sinograms

    Args:
        tomogram: three-dimensional stack of sinograms 
        dic: input dictionary 
        
    Returns:
        (ndarray, ndarray): stacking 3D reconstructed volume, reconstructed sinograms  

    * CPU function
    * This function uses a shared array through package 'SharedArray'.
    * The total number of images will be divided by the number of processes given as input
    * SharedArray names are provided by uuid.uuid4() 
    
    Parameters:
    
    * ``dic['nangles']``:  Number of angles
    * ``dic['angles']``:  arrya of angles
    * ``dic['gpu']``:  List of GPU devices used for computation 
    * ``dic['blocksize']``:  Number of images to compute parallel radon transform 
    * ``dic['niterations']``:  Tuple for number of iterations. First position refers to the global number of
      iterations for the {Transmission,Emission}/EM algorithms. Second and Third positions refer to the local 
      number of iterations for EM/TV number of iterations, as indicated by Yan & Luminita`s article.
    * ``dic['regularization']``:  Regularization parameter for EM/TV
    * ``dic['epsilon']``:  Smoothness for the TV operator
    * ``dic['method']``:  EM-method type ``eEM``, ``tEM`` or ``EM/TV``
    
       #. ``eEM`` refers to the  Emission Expectation Maximization Algorithm, where we solve
          :math:`Ax = b`, being :math:`A` the discretized version for the Radon transform and 
          :math:`b` the input sinograms in the parallel geometry.

       #. ``tEM`` refers to the Transmission Expectation Maximization Algorithm, where we solve
          :math:`Ax = b` using :math:`exp(-b)` as the photon count and 1's as the flat-field 
          measurement. 

       #. ``EMTV`` refers to the combination of ``eEM`` and a Total variation step, as indicated
          in the manuscript of Yan & Luminita ``DOI:10.1117/12.878238``.

    """
    
    nslices   = tomogram.shape[0]
    nrays     = tomogram.shape[2]
    nangles   = dic['nangles']
    angles    = dic['angles']
    gpus      = dic['gpu']
    blocksize = dic['blocksize']
    niter     = dic['niterations']
    reg       = dic['regularization']
    eps       = dic['epsilon']
    method    = dic['method']
    
    if blocksize > nslices // len(gpus):
        print('ssc-raft: Error! Please check block size!')
    
    name1 = str( uuid.uuid4())
    
    try:
        sa.delete(name1)
    except:
        pass
        
    output1  = sa.create(name1,[nslices, nrays, nrays], dtype=numpy.float32)

    _params_ = ( output1, tomogram, nslices, nangles, gpus, blocksize, niter, reg, eps, method, angles)
    
    _build_em_mpfs_( _params_ )

    sa.delete(name1)
    
    return output1
