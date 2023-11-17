from ....rafttypes import *

from ctypes import c_int as int32
from ctypes import c_float as float32
from ctypes import POINTER
from ctypes import c_void_p  as void_p

import numpy
# import uuid
# import SharedArray as sa
import time

def _emfreq_GPU_(count, flat, angles, gpus, zpad, interpolation, dx, niter):

    nslices = count.shape[0]
    nangles = count.shape[1]
    nrays   = count.shape[2]
    
    recon   = numpy.zeros([nslices, nrays, nrays], dtype=numpy.float32)
    recon_p = recon.ctypes.data_as(void_p)

    count   = CNICE(count) #sino pointer
    count_p = count.ctypes.data_as(void_p) 

    flat    = CNICE(flat) #sino pointer
    flat_p  = flat.ctypes.data_as(void_p) 
    
    try:
        ang   = CNICE(angles) #angles pointer
        ang_p = ang.ctypes.data_as(void_p) 
    except:
        angles = numpy.linspace(0,numpy.pi,nangles,endpoint=True)

        ang   = CNICE(angles) #angles pointer
        ang_p = ang.ctypes.data_as(void_p) 

    libraft.emfreqgpu( count_p, recon_p, ang_p, flat_p, 
                    int32(nrays), int32(nangles), int32(nslices),
                    int32(zpad),  int32(interpolation), float32(dx), int32(niter), int32(gpus))

    return recon

def _emfreq_multiGPU_(count, flat, angles, gpus, zpad, interpolation, dx, niter):

    nslices = count.shape[0]
    nangles = count.shape[1]
    nrays   = count.shape[2]
    
    ngpus   = len(gpus)
    gpus    = numpy.array(gpus)
    gpus    = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_p  = gpus.ctypes.data_as(void_p)

    recon   = numpy.zeros([nslices, nrays, nrays], dtype=numpy.float32)
    recon_p = recon.ctypes.data_as(void_p)

    count   = CNICE(count) #sino pointer
    count_p = count.ctypes.data_as(void_p) 

    flat    = CNICE(flat) #sino pointer
    flat_p  = flat.ctypes.data_as(void_p) 
    
    try:
        ang   = CNICE(angles) #angles pointer
        ang_p = ang.ctypes.data_as(void_p) 
    except:
        angles = numpy.linspace(0,numpy.pi,nangles,endpoint=True)

        ang   = CNICE(angles) #angles pointer
        ang_p = ang.ctypes.data_as(void_p) 

    libraft.emfreqblock( count_p, recon_p, ang_p, flat_p, 
                    int32(nrays), int32(nangles), int32(nslices),
                    int32(zpad),  int32(interpolation), float32(dx), 
                    int32(niter), int32(ngpus), gpus_p)

    return recon

def emfreq(count, flat, dic, **kwargs):
    """ Expectation maximization for 3D tomographic parallel sinograms

    Args:
        tomogram: three-dimensional stack of sinograms (slices,angles,rays) 
        dic: input dictionary 
        
    Returns:
        (ndarray): stacking 3D reconstructed volume, reconstructed sinograms (z,y,x)

    * CPU function
    * This function uses a shared array through package 'SharedArray'.
    * The total number of images will be divided by the number of processes given as input
    * SharedArray names are provided by uuid.uuid4() 
    
    Parameters:
    
    * ``dic['nangles']``:  Number of angles
    * ``dic['angles']``:  list of angles
    * ``dic['gpu']``:  List of GPU devices used for computation 
    * ``dic['blocksize']``:  Number of images to compute parallel radon transform 
    * ``dic['niterations']``:  Tuple for number of iterations. First position refers to the global number of
      iterations for the {Transmission,Emission}/EM algorithms. Second and Third positions refer to the local 
      number of iterations for EM/TV number of iterations, as indicated by Yan & Luminita`s article.
    * ``dic['regularization']``:  Regularization parameter for EM/TV
    * ``dic['epsilon']``:  Smoothness for the TV operator
    * ``dic['is360']`` (bool): It is used if no ``dic['angles']`` is set. True: 360 degrees acquisition; False: 180 degrees acquisition.
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
    # Set default dictionary parameters:

    # dicparams = ('gpu','angles','nangles','niterations','regularization','epsilon','method','is360','blocksize')
    # defaut    = ([0],None,tomogram.shape[1],[8,3,8],1e-3,1e-1,'eEM',False, 1)

    # SetDictionary(dic,dicparams,defaut)

    # Set parameters

    gpus          = dic['gpu']
    angles        = numpy.asarray(dic['angles'])
    zpad          = dic['zpad']
    interpolation = dic['interpolation']
    dx            = dic['dx']
    niter         = dic['niterations'][0]

    if len(gpus) == 1:

        gpu    = gpus[0]
        output = _emfreq_GPU_(count, flat, angles, gpu, zpad, interpolation, dx, niter)

    else:
        output = _emfreq_multiGPU_(count, flat, angles, gpus, zpad, interpolation, dx, niter)
        pass

    # Garbage Collector
    # lists are cleared whenever a full collection or
    # collection of the highest generation (2) is run
    # collected = gc.collect() # or gc.collect(2)
    # logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')

    return output