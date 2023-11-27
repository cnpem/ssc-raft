# Authors: Paola Ferraz, Giovanni L. Baraldi, Gilberto Martinez

from ..rafttypes import *
import numpy as np
import gc
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t


def flatdarkGPU(frames, flat, dark, dic):
        
        gpus       = dic['gpu']     
        ngpus      = len(gpus)

        gpus       = numpy.array(gpus)
        gpus       = np.ascontiguousarray(gpus.astype(np.intc))
        gpusptr    = gpus.ctypes.data_as(void_p)

        is_log     = dic['uselog']
        nrays      = frames.shape[-1]
        nangles    = frames.shape[-2]
        
        if is_log:
                is_log = 1
                logger.info(f'Returning Correction with log() applied.')
        else:
                is_log = 0
                logger.info(f'Returning Correction without log() applied.')

        if len(frames.shape) == 2:
                nslices = 1
        elif len(frames.shape) == 3:
                nslices = frames.shape[0]
        else:
                logger.error(f'Incorrect data dimension! It accepts only 2- or 3-dimension array. ')
                logger.error(f'Data dimension is ({frames.shape}).')
                logger.error(f'Finishing run...')
                sys.exit(1)
                
        if len(flat.shape) == 2:
                nflats  = 1
        elif len(flat.shape) == 3:
                nflats = flat.shape[-2]
        else:
                logger.error(f'Incorrect flat dimension! It accepts only 2- or 3-dimension array. ')
                logger.error(f'Flat dimension is ({flat.shape}).')
                logger.error(f'Finishing run...')
                sys.exit(1)
        
        if len(dark.shape) == 2:
                dark = np.expand_dims(dark,1)
        elif len(dark.shape) == 3:
                dark = dark[0]
                dark = np.expand_dims(dark,1)
        else:
                logger.error(f'Incorrect dark dimension! It accepts only 2- or 3-dimension array. ')
                logger.error(f'Dark dimension is ({dark.shape}).')
                logger.error(f'Finishing run...')
                sys.exit(1)

        logger.info(f'Number of flats is {nflats}.')

        if flat.shape[-1] != nrays or flat.shape[-2] != nslices:
                logger.error(f'Flat dimension ({flat.shape[-1]},{flat.shape[-2]}) does not match data dimension ({frames.shape[-1]},{frames.shape[-2]}).')
                logger.error(f'Finishing run...')
                sys.exit(1)

        if dark.shape[-1] != nrays or dark.shape[-2] != nslices:
                logger.error(f'Dark dimension ({dark.shape[-1]},{dark.shape[-2]}) does not match data dimension ({frames.shape[-1]},{frames.shape[-2]}).')
                logger.error(f'Finishing run...')
                sys.exit(1)

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
        is_log     = int32(is_log)
        
        libraft.getFlatDarkMultiGPU(gpusptr, int32(ngpus), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats, is_log)

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
                flat   (ndarray): Flat of size [number of flats, slices, rays]
                dark   (ndarray): Dark of size [slices, rays]
                dic (dictionary): Dictionary with the parameters info.

        Returns:
                ndarray: Corrected frames (or projections) of dimension [slices, angles, rays]
        
        Dictionary parameters:
                *``experiment['gpu']`` (int list): List of GPUs.
                *``experiment['uselog']`` (bool): Apply logarithm or not.
        """        
        logger.info(f'Begin Flat and Dark correction.')

        dicparams = ('gpu','uselog')
        defaut    = ([0],True)
        
        SetDictionary(dic,dicparams,defaut)

        frames = flatdarkGPU( frames, flat, dark, dic ) 

        # Garbage Collector
        # lists are cleared whenever a full collection or
        # collection of the highest generation (2) is run
        collected = gc.collect() # or gc.collect(2)
        logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')


        return frames