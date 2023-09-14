# Authors: Giovanni L. Baraldi, Gilberto Martinez

from ..rafttypes import *
import numpy as np
import gc
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t


def flatdarkMultiGPU(frames, flat, dark, dic):
        
        gpus       = dic['gpu']     
        ngpus      = len(gpus)

        gpus       = numpy.array(gpus)
        gpus       = np.ascontiguousarray(gpus.astype(np.intc))
        gpusptr    = gpus.ctypes.data_as(void_p)

        is_log     = dic['uselog']
        nrays      = frames.shape[-1]
        nangles    = frames.shape[0]
        Tframes    = nangles
        
        if is_log:
                is_log = 1
        else:
                is_log = 0

        if len(frames.shape) == 2:
                nslices = 1
        else:
                nslices = frames.shape[-2]

        if len(flat.shape) == 2:
                flat = np.expand_dims(flat,0)
                nflats  = 1
        else:
                nflats = flat.shape[0]

        if len(dark.shape) == 2:
                dark = np.expand_dims(dark,0)

        # Change Frames order from [angles,slices,rays] to [slices,angles,rays] for easier computation
        frames    = np.swapaxes(frames,0,1)
        
        # Change Flat order from [number of flats,slices,rays] to [slices,number of flats,rays] for easier computation
        flat      = np.swapaxes(flat,0,1)

        # Change Dark order from [1,slices,rays] to [slices,1,rays] for easier computation
        dark      = np.swapaxes(dark,0,1)

        logger.info(f'Number of flats is {nflats}.')
        if nflats > 1:
                logger.info(f'Interpolating flats before and after.')

        logger.info(f'Flat dimension is ({flat.shape}) = (slices,number of flats,rays).')
        logger.info(f'Dark dimension is ({dark.shape}) = (slices,number of darks,rays).')

        flat      = np.ascontiguousarray(flat.astype(np.float32))
        flatptr   = flat.ctypes.data_as(void_p)

        dark      = np.ascontiguousarray(dark.astype(np.float32))
        darkptr   = dark.ctypes.data_as(void_p)
        
        frames    = np.ascontiguousarray(frames.astype(np.float32))
        framesptr = frames.ctypes.data_as(void_p)

        nrays      = int32(nrays)
        nangles    = int32(nangles)
        nslices    = int32(nslices)
        nflats     = int32(nflats)
        Tframes    = int32(Tframes)
        Initframes = int32(0)
        is_log     = int32(is_log)
        
        libraft.flatdark_block(gpusptr, int32(ngpus), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats, Tframes, Initframes, is_log)

        return frames 


def flatdarkGPU(frames, flat, dark, dic):
        
        gpu        = dic['gpu'][0]     
        is_log     = dic['uselog']
        nrays      = frames.shape[-1]
        nangles    = frames.shape[0]
        Tframes    = nangles
        
        if is_log:
                is_log = 1
        else:
                is_log = 0

        if len(frames.shape) == 2:
                nslices = 1
        else:
                nslices = frames.shape[-2]

        if len(flat.shape) == 2:
                flat = np.expand_dims(flat,0)
                nflats  = 1
        else:
                nflats = flat.shape[0]
        
        if len(dark.shape) == 2:
                dark = np.expand_dims(dark,0)
        
        # Change Frames order from [angles,slices,rays] to [slices,angles,rays] for easier computation
        frames    = np.swapaxes(frames,0,1)
        
        # Change Flat order from [number of flats,slices,rays] to [slices,number of flats,rays] for easier computation
        flat      = np.swapaxes(flat,0,1)

        # Change Dark order from [1,slices,rays] to [slices,1,rays] for easier computation
        dark      = np.swapaxes(dark,0,1)

        logger.info(f'Number of flats is {nflats}.')
        if nflats > 1:
                logger.info(f'Interpolating flats before and after.')

        logger.info(f'Flat dimension is ({flat.shape}) = (slices,number of flats,rays).')
        logger.info(f'Dark dimension is ({dark.shape}) = (slices,number of darks,rays).')

        flat      = np.ascontiguousarray(flat.astype(np.float32))
        flatptr   = flat.ctypes.data_as(void_p)

        dark      = np.ascontiguousarray(dark.astype(np.float32))
        darkptr   = dark.ctypes.data_as(void_p)
        
        frames    = np.ascontiguousarray(frames.astype(np.float32))
        framesptr = frames.ctypes.data_as(void_p)

        nrays      = int32(nrays)
        nangles    = int32(nangles)
        nslices    = int32(nslices)
        nflats     = int32(nflats)
        Tframes    = int32(Tframes)
        Initframes = int32(0)
        is_log     = int32(is_log)
        
        libraft.flatdark_gpu(int32(gpu), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats, Tframes, Initframes, is_log)

        return frames 


def correct_projections(frames, flat, dark, dic, **kwargs):
        """ Function to correct tomography projections (or frames) with flat and dark. 
        
        Can be computed in two ways.

        .. math::
        T = \log{- \frac{D - D_d}{D_f - D_d}}

        for transmission tomography, and

        .. math::
        T = \frac{D - D_d}{D_f - D_d}

        for phase contrast tomography. Where :math:`T` is the corrected tomogram, :math:`D` is the projection volume, 
        :math:`D_f` is the flat projections and :math:`D_d` is the dark projections

        Args:
            frames (ndarray): Frames (or projections) of size [angles, slices, rays]
            flat (ndarray): Flat of size [number of flats, slices, rays]
            dark (ndarray): Dark of size [slices, rays]
            dic (dictionary): Dictionary with the parameters info.

        Returns:
            ndarray: Corrected frames (or projections)of  size [slices, angles, rays]
        
         Dictionary parameters:
                *``experiment['gpu']`` (int list): List of GPUs.
                *``experiment['uselog']`` (bool): Apply logarithm or not.
        """        
        logger.info(f'Begin Flat and Dark correction.')

        dicparams = ('gpu','uselog')
        defaut    = ([0],True)
        
        SetDictionary(dic,dicparams,defaut)

        gpus = dic['gpu']

        if len(gpus) == 1:
                output = flatdarkGPU( frames, flat, dark, dic )
        else:
                output = flatdarkMultiGPU( frames, flat, dark, dic ) 

        # if dic['uselog']:
        #         output[np.isinf(output)] = 0.0
        #         output[np.isnan(output)] = 0.0
        # else:
        #         output[np.isinf(output)] = 1.0
        #         output[np.isnan(output)] = 1.0

        # Garbage Collector
        # lists are cleared whenever a full collection or
        # collection of the highest generation (2) is run
        # collected = gc.collect() # or gc.collect(2)
        # logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')


        return output