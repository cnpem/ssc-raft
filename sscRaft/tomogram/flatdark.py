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

        if len(flat.shape) == 2:
                nflats  = 1
        else:
                nflats = flat.shape[0]

        flat = np.ascontiguousarray(flat.astype(np.float32))
        flatptr = flat.ctypes.data_as(void_p)

        dark = np.ascontiguousarray(dark.astype(np.float32))
        darkptr = dark.ctypes.data_as(void_p)
        
        frames = np.ascontiguousarray(frames.astype(np.float32))
        framesptr = frames.ctypes.data_as(void_p)

        nrays   = int32(nrays)
        nangles = int32(nangles)
        nslices = int32(nslices)
        nflats  = int32(nflats)
        
        if dic['uselog']:
                libraft.flatdark_log_block(gpusptr, int32(ngpus), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats)
        else:
                libraft.flatdark_block(gpusptr, int32(ngpus), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats)

        return frames 


def flatdarkGPU(frames, flat, dark, dic):
        
        gpus = dic['gpu']     
        ngpus = len(gpus)

        nrays   = frames.shape[-1]
        nangles = frames.shape[0]
        
        if len(frames.shape) == 2:
                nslices = 1
        else:
                nslices = frames.shape[-2]

        if len(flat.shape) == 2:
                nflats  = 1
        else:
                nflats = flat.shape[0]

        flat = np.ascontiguousarray(flat.astype(np.float32))
        flatptr = flat.ctypes.data_as(void_p)

        dark = np.ascontiguousarray(dark.astype(np.float32))
        darkptr = dark.ctypes.data_as(void_p)
        
        frames = np.ascontiguousarray(frames.astype(np.float32))
        framesptr = frames.ctypes.data_as(void_p)
        
        nrays   = int32(nrays)
        nangles = int32(nangles)
        nslices = int32(nslices)
        nflats  = int32(nflats)

        if dic['uselog']:
                libraft.flatdark_log_gpu(int32(ngpus), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats)
        else:
                libraft.flatdark_gpu(int32(ngpus), framesptr, flatptr, darkptr, nrays, nslices, nangles, nflats)

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

        dicparams = ('gpu','uselog')
        defaut = ([0],True)
        
        SetDictionary(dic,dicparams,defaut)

        gpus = dic['gpu']

        if len(gpus) == 1:
                output = flatdarkGPU( frames, flat, dark, dic )
        else:
                output = flatdarkMultiGPU( frames, flat, dark, dic ) 

        return np.swapaxes(output,0,1)