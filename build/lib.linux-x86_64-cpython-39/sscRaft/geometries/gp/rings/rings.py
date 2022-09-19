from ....rafttypes import *
import numpy as np
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t
import uuid
import SharedArray as sa

def RingsMultiGPU(tomogram, dic):
        """Apply rings correction on tomogram by blocks of rings

        Args:
            tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
            lambdareg (int, optional): Lambda regularization of rings. Defaults to -1 (automatic).
            ringblocks (int, optional): Blocks of rings to be used. Defaults to 2.
            gpus (list, optional): List of gpus. Defaults to [0].

        Returns:
            ndarray: tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        """   

        gpus = dic['gpu']     
        ngpus = len(gpus)

        gpus = numpy.array(gpus)
        gpus = np.ascontiguousarray(gpus.astype(np.int32))
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

        return tomogram


def RingsGPU(tomogram, dic, gpu = 0):
        """Apply rings correction on tomogram by blocks of rings

        Args:
            tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
            lambdareg (int, optional): Lambda regularization of rings. Defaults to -1 (automatic).
            ringblocks (int, optional): Blocks of rings to be used. Defaults to 2.
            gpus (list, optional): List of gpus. Defaults to [0].

        Returns:
            ndarray: tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
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

        return tomogram


def _worker_rings_(params, start, end, gpu, process):

        output = params[0]
        data   = params[1]
        dic    = params[6]

        logger.info(f'Applying Rings: begin process {process} on gpu {gpu}')

        output[start:end,:,:] = RingsGPU( data[start:end, :, :], dic, gpu )

        logger.info(f'Applying Rings: end process {process} on gpu {gpu}')
    

def _build_rings_(params):
 
        nslices = params[2]
        gpus    = params[5]
        ngpus = len(gpus)

        b = int( numpy.ceil( nslices/ngpus )  ) 

        processes = []
        for process in range( ngpus ):
                begin_ = process*b
                end_   = min( (process+1)*b, nslices )

                p = multiprocessing.Process(target=_worker_rings_, args=(params, begin_, end_, gpus[process], process))
                processes.append(p)
    
        for p in processes:
                p.start()

        for p in processes:
                p.join()


def rings_gpublock( tomogram, dic ):

        nslices     = tomogram.shape[0]
        nangles     = tomogram.shape[1]
        nrays       = tomogram.shape[2]
        gpus        = dic['gpu']

        name = str(uuid.uuid4())

        try:
                sa.delete(name)
        except:
                pass

        output  = sa.create(name,[nslices, nangles, nrays], dtype=np.float32)
        print(output.dtype)

        _params_ = ( output, tomogram, nslices, nangles, nrays, gpus, dic)

        _build_rings_( _params_ )

        sa.delete(name)

        return output


def rings_thread(tomogram, dic, **kwargs):

        dicparams = ('gpu','lambda rings','rings block')
        defaut = ([0],0,2)

        SetDictionary(dic,dicparams,defaut)

        gpus = dic['gpu']
        
        if len(gpus) == 1:
                gpu = gpus[0]
                output = RingsGPU( tomogram, dic, gpu )
        else:
                output = rings_gpublock( tomogram, dic ) 

        return output

def rings(tomogram, dic, **kwargs):

        dicparams = ('gpu','lambda rings','rings block')
        defaut = ([0],0,2)

        SetDictionary(dic,dicparams,defaut)

        gpus = dic['gpu']
        
        if len(gpus) == 1:
                gpu = gpus[0]
                output = RingsGPU( tomogram, dic, gpu )
        else:
                output = RingsMultiGPU( tomogram, dic ) 

        return output