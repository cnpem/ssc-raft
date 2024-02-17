from ...rafttypes import *

import numpy
import time

def _eEMRT_GPU_(tomo, angles, iterations, gpus):
    # MultiGPU withou semafaros

    if len(tomo.shape) == 2:
        nslices = 1
    else:
        nslices = tomo.shape[0]

    nangles       = tomo.shape[1]
    nrays         = tomo.shape[2]

    objsize       = tomo.shape[2]
    
    ngpus         = len(gpus)
    gpus          = numpy.array(gpus)
    gpus          = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr      = gpus.ctypes.data_as(ctypes.c_void_p)
    
    obj           = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
    obj_ptr       = obj.ctypes.data_as(ctypes.c_void_p)

    tomo          = CNICE(tomo) #sino pointer
    tomo_ptr      = tomo.ctypes.data_as(ctypes.c_void_p) 
    
    angles        = numpy.array(angles)
    angles        = CNICE(angles) #angles pointer
    angles_ptr    = angles.ctypes.data_as(ctypes.c_void_p) 

    param_int     = [nrays, nangles, nslices, objsize, iterations]
    param_int     = numpy.array(param_int)
    param_int     = CNICE(param_int,numpy.int32)
    param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

    param_float     = [0]
    param_float     = numpy.array(param_float)
    param_float     = CNICE(param_float,numpy.float32)
    param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)

    libraft.get_eEM_RT_MultiGPU(gpus_ptr, ctypes.c_int(ngpus),
                    obj_ptr, tomo_ptr, angles_ptr, 
                    param_float_ptr, param_int_ptr)

    return obj

def _tEMRT_GPU_(counts, flat, angles, iterations, gpus):
    # MultiGPU withou semafaros

    if len(counts.shape) == 2:
        nslices = 1
    else:
        nslices = counts.shape[0]

    nangles       = counts.shape[1]
    nrays         = counts.shape[2]

    objsize       = counts.shape[2]

    # counts        = numpy.exp(-counts)
    # flat          = numpy.ones(counts.shape)
    
    ngpus         = len(gpus)
    gpus          = numpy.array(gpus)
    gpus          = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr      = gpus.ctypes.data_as(ctypes.c_void_p)
    
    obj           = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
    obj_ptr       = obj.ctypes.data_as(ctypes.c_void_p)

    counts          = CNICE(counts) 
    counts_ptr      = counts.ctypes.data_as(ctypes.c_void_p) 

    flat          = CNICE(flat) 
    flat_ptr      = flat.ctypes.data_as(ctypes.c_void_p) 
    
    angles        = numpy.array(angles)
    angles        = CNICE(angles) #angles pointer
    angles_ptr    = angles.ctypes.data_as(ctypes.c_void_p) 

    param_int     = [nrays, nangles, nslices, objsize, iterations]
    param_int     = numpy.array(param_int)
    param_int     = CNICE(param_int,numpy.int32)
    param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

    param_float     = [0]
    param_float     = numpy.array(param_float)
    param_float     = CNICE(param_float,numpy.float32)
    param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)

    libraft.get_tEM_RT_MultiGPU(gpus_ptr, ctypes.c_int(ngpus),
                    obj_ptr, counts_ptr, flat_ptr, angles_ptr, 
                    param_float_ptr, param_int_ptr)

    return obj


def _tEMFQ_GPU_(count, flat, angles, 
                     pad, interpolation, det_pixel, tv_reg, iterations, 
                     gpus, obj=None):

    if len(count.shape) == 2:
        nslices = 1
    else:
        nslices = count.shape[0]

    nangles     = count.shape[1]
    nrays       = count.shape[2]
    
    ngpus       = len(gpus)
    gpus        = numpy.array(gpus)
    gpus        = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr    = gpus.ctypes.data_as(ctypes.c_void_p)

    # obj can be the initial guess
    if obj is not None:
        obj     = CNICE(obj)
        obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)
    else:
        obj     = numpy.ones([nslices, nrays, nrays], dtype=numpy.float32)
        obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    count       = CNICE(count) 
    count_ptr   = count.ctypes.data_as(ctypes.c_void_p) 

    flat        = CNICE(flat) 
    flat_ptr    = flat.ctypes.data_as(ctypes.c_void_p) 
    
    angles      = numpy.array(angles)
    angles      = CNICE(angles) 
    angles_ptr  = angles.ctypes.data_as(ctypes.c_void_p) 


    libraft.get_tEM_FQ_MultiGPU( gpus_ptr, ctypes.c_int(ngpus),
                    count_ptr, obj_ptr, angles_ptr, flat_ptr,
                    ctypes.c_int(nrays), ctypes.c_int(nangles), ctypes.c_int(nslices),
                    ctypes.c_int(pad),  ctypes.c_int(interpolation), ctypes.c_float(det_pixel), 
                    ctypes.c_float(tv_reg), ctypes.c_int(iterations))

    return obj


def em(data, dic, flat = None, angles = None, guess = None, **kwargs):
    """ Expectation maximization (EM) for 3D tomographic reconstructions for parallel, 
    conebeam and fanbeam geometries.

    Implemented methods for parallel geometry:

        * ``eEMRT``: Emission EM using Ray Tracing as forward and inverse operators.
        * ``tEMRT``: Transmission EM using Ray Tracing as forward and inverse operators.
        * ``tEMFQ``: Transmission EM using the Fourier Slice Theorem (FST) for the forward operator and Backprojection Slice Theorem (BST) for the inverse operator.

    Args:
        data (ndarray): Tomographic 3D data. The axis are (slices,angles,rays) 
        dic (dict): input dictionary 
        
    Returns:
        (ndarray): stacking 3D reconstructed volume, reconstructed sinograms (z,y,x)

    * One or MultiGPUs. 
    * Calls function ``_emfreq_GPU_()``.
    * Calls function ``_emfreq_multiGPU_()``.
    * Calls function ``_emfreq_multiGPU_()``.
    
    Dictionary parameters:
    
        * ``dic['gpu']`` (int list):  List of GPU devices used for computation [required]
        * ``dic['flat']`` (ndarray):  Flat 2D data. Tha axis are (slices,rays)  [required]
        * ``dic['angles[rad]']`` (floar list):  List of angles in radians [required]
        * ``dic['detectorPixel[m]']`` (float): Detector pixel size in meters [required for ``tEMFQ``]
        * ``dic['method']`` (str): Choose EM-method [required]
    
            #. ``eEMRT``: Emission EM using Ray Tracing as forward and inverse operators.

            #. ``tEMRT``: Transmission EM using Ray Tracing as forward and inverse operators.

            #. ``tEMFQ``: Transmission EM using the Fourier Slice Theorem (FST) for the forward operator and Backprojection Slice Theorem (BST) for the inverse operator.

        * ``dic['iterations']`` (int, optional): Global number of iterations [default: 100]
        * ``dic['interpolation']`` (str, optional):  Type of interpolation. Options: \'nearest\' or \'bilinear\' [default: \'bilinear\']
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 2]  

    """
    # Set default dictionary parameters:

    dicparams = ('iterations','detectorPixel[m]','padding')
    defaut    = (100,0.0,2)

    SetDictionary(dic,dicparams,defaut)

    gpus          = dic['gpu']
    method        = dic['method']

    iterations    = dic['iterations']
    TV_iterations = 0 #dic['TV iterations']

    det_pixel     = dic['detectorPixel[m]']
    pad           = dic['padding']

    # Regularization and smoothness parameter for the TV (Total Variation method)
    tv_reg        = 0.0 # dic['regularization'] 
    tv_smooth     = 0.0 # dic['smoothness']

    # Interpolation for EM Frequency
    interpolation = setInterpolation(dic['interpolation'])

    try:
        angles = dic['angles[rad]']
    except:
        if angles is None:
            logger.error(f'Missing angles list!! Finishing run...') 
            raise ValueError(f'Missing angles list!!')
    try:
        flat = dic['flat']
    except:
        if flat is None:
            flat = numpy.ones((data.shape[0],data.shape[2])) 

    if method == 'eEMRT':

        output = _eEMRT_GPU_(data, angles, iterations, gpus) 
    
    elif method == 'tEMRT':

        output = _tEMRT_GPU_(data, flat, angles, iterations, gpus)

    elif method == 'tEMFQ':

        output = _tEMFQ_GPU_(data, flat, angles, 
                    pad, interpolation, det_pixel, tv_reg, iterations, 
                    gpus, guess)  

    else:
        logger.error(f'Invalid EM method:{method}')
        raise ValueError(f'Invalid EM method:{method}')

    return output


