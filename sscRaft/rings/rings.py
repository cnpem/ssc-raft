from ..rafttypes import *
import numpy as np
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t
import uuid
import SharedArray as sa

def RingsMultiGPU(tomogram, dic):
        """Apply rings correction on tomogram by blocks of rings in MULTIGPU

        Args:
                tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
                dic (dictionary): Dictionary with the parameters.
                gpu (int): GPU to use. Defaults to 0.

        Returns:
                ndarray: tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        
        Dictionary parameters:
                *``dic['lambda rings']`` (float): Lambda regularization of rings. Value 0 is automatic.
                *``dic['rings block']`` (int): Blocks of rings to be used.    
                *``dic['gpu']`` (int list): List of GPUs to use. 
        """   

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

        lambdarings = dic['lambda rings']
        ringsblock  = dic['rings block']

        if lambdarings == 0:
                logger.warning(f'No Rings regularization: Lambda regularization is set to {lambdarings}.')
                return tomogram
        elif lambdarings < 0:
                logger.info(f'Using automatic Rings removal')
        else:
                logger.info(f'Rings removal with Lambda regularization of {lambdarings}')  

        tomogram = np.ascontiguousarray(tomogram.astype(np.float32))
        tomogramptr = tomogram.ctypes.data_as(void_p)
        
        start = time()

        libraft.ringsblock(gpusptr, int32(ngpus), tomogramptr, int32(nrays), int32(nangles), int32(nslices), float32(lambdarings), int32(ringsblock))
        
        elapsed = time() - start
        print('Rings Time', elapsed)

        return tomogram


def RingsGPU(tomogram, dic, gpu = 0):
        """Apply rings correction on tomogram by blocks of rings in GPU

        Args:
                tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
                dic (dictionary): Dictionary with the parameters.
                gpu (int): GPU to use. Defaults to 0.

        Returns:
                ndarray: tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        
        Dictionary parameters:
                *``dic['lambda rings']`` (float): Lambda regularization of rings. Value 0 is automatic.
                *``dic['rings block']`` (int): Blocks of rings to be used. 
        """        

        nrays   = tomogram.shape[-1]
        nangles = tomogram.shape[-2]

        if len(tomogram.shape) == 2:
                nslices = 1
        else:
                nslices = tomogram.shape[0]

        lambdarings = dic['lambda rings']
        ringsblock  = dic['rings block']

        if lambdarings == 0:
                logger.warning(f'No Rings regularization: Lambda regularization is set to {lambdarings}.')
                return tomogram
        elif lambdarings < 0:
                logger.info(f'Using automatic Rings removal')
        else:
                logger.info(f'Rings removal with Lambda regularization of {lambdarings}')  

        tomogram = np.ascontiguousarray(tomogram.astype(np.float32))
        tomogramptr = tomogram.ctypes.data_as(void_p)
        
        start = time()

        libraft.ringsgpu(int32(gpu), tomogramptr, int32(nrays), int32(nangles), int32(nslices), float32(lambdarings), int32(ringsblock))
        
        elapsed = time() - start
        print('Rings Time', elapsed)

        return tomogram


def rings(tomogram, dic, **kwargs):
        """Apply rings correction on tomogram by blocks of rings in MultiGPU

        Args:
                tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
                dic (dictionary): Dictionary with the parameters.

        Returns:
                ndarray: tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        
        Dictionary parameters:
                *``dic['lambda rings']`` (float): Lambda regularization of rings. Defaults to 0 (automatic).
                *``dic['rings block']`` (int): Blocks of rings to be used. Defaults to 2.    
                *``dic['gpu']`` (int list): List of GPUs to use. Defaults to [0].
        """

        dicparams = ('gpu','lambda rings','rings block')
        defaut = ([0],0,2)

        SetDictionary(dic,dicparams,defaut)

        gpus = dic['gpu']
        
        if len(gpus) == 1:
                gpu = gpus[0]
                output = RingsGPU( tomogram, dic, gpu )
        else:
                output = RingsMultiGPU( tomogram, dic ) 

        # Garbage Collector
        # lists are cleared whenever a full collection or
        # collection of the highest generation (2) is run
        # collected = gc.collect() # or gc.collect(2)
        # logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')

        return output