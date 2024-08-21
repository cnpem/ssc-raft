from ...rafttypes import *

def eEMRT_GPU_(tomo, angles, iterations, gpus, blocksize, obj = None):
    """ Wrapper for MultiGPU/CUDA function of the 
    Emission Expectation maximization (EM) method for 3D tomographic reconstructions in 
    parallel beam geometry. 

    This EM method uses Ray Tracing as forward and inverse operators.

    Args:
        tomo (ndarray): Tomographic 3D data with shape (slices,angles,lenght) 
        angles (float list): List of angles in radians
        iterations (int): EM iterations
        gpus (int list): List of gpus
        
    Returns:
        (ndarray): stacking 3D reconstructed volume (z,y,x) or 2D reconstructed sinograms (y,x)
    """
    # MultiGPU without semafaros

    if len(tomo.shape) == 2:
        nslices = 1
    else:
        nslices = tomo.shape[0]

    nangles       = tomo.shape[-2]
    nrays         = tomo.shape[-1]

    objsize       = tomo.shape[-1]
    
    ngpus         = len(gpus)
    gpus          = numpy.array(gpus)
    gpus          = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr      = gpus.ctypes.data_as(ctypes.c_void_p)
    
    if obj is not None:
        obj     = CNICE(obj)
        obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)
    else:
        obj     = numpy.ones([nslices, objsize, objsize], dtype=numpy.float32)
        obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    tomo          = CNICE(tomo) #sino pointer
    tomo_ptr      = tomo.ctypes.data_as(ctypes.c_void_p) 
    
    angles        = numpy.array(angles)
    angles        = CNICE(angles) #angles pointer
    angles_ptr    = angles.ctypes.data_as(ctypes.c_void_p) 

    padx, pady, padz = 0,0,0
    nflats           = 1

    param_int     = [nrays, nangles, nslices, objsize, 
                     padx, pady, padz, nflats, iterations, blocksize]
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

def tEMRT_GPU_(counts, flat, angles, iterations, gpus, blocksize, obj = None):
    """ Wrapper for MultiGPU/CUDA function of the 
    Transmission Expectation maximization (EM) method for 3D tomographic reconstructions in 
    parallel beam geometry. Flat (or empty) here is defined by a measurement without
    a sample, to measure the background.

    This EM method uses Ray Tracing as forward and inverse operators.

    Args:
        counts (ndarray): Photon counts 3D data with shape (slices,angles,lenght) 
        flat (ndarray): Flat (or background) data with shape (slices,1,lenght) 
        angles (float list): List of angles in radians
        iterations (int): EM iterations
        gpus (int list): List of gpus
        
    Returns:
        (ndarray): stacking 3D reconstructed volume (z,y,x) or 2D reconstructed sinograms (y,x)
    """
    # MultiGPU withou semafaros

    if len(counts.shape) == 2:
        nslices = 1
    else:
        nslices = counts.shape[0]

    if len(counts.shape) == 2:
        nflats = 1
    else:
        nflats = flat.shape[0]

    nangles       = counts.shape[-2]
    nrays         = counts.shape[-1]

    objsize       = counts.shape[-1]

    counts        = numpy.exp(-counts, counts)
    # flat          = numpy.ones(counts.shape)
    
    ngpus         = len(gpus)
    gpus          = numpy.array(gpus)
    gpus          = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr      = gpus.ctypes.data_as(ctypes.c_void_p)

    if obj is not None:
        obj     = CNICE(obj)
        obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)
    else:
        obj     = numpy.ones([nslices, objsize, objsize], dtype=numpy.float32)
        obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    counts        = CNICE(counts) 
    counts_ptr    = counts.ctypes.data_as(ctypes.c_void_p) 

    flat          = CNICE(flat) 
    flat_ptr      = flat.ctypes.data_as(ctypes.c_void_p) 
    
    angles        = numpy.array(angles)
    angles        = CNICE(angles) #angles pointer
    angles_ptr    = angles.ctypes.data_as(ctypes.c_void_p) 

    padx, pady, padz = 0,0,0

    param_int     = [nrays, nangles, nslices, objsize, 
                     padx, pady, padz, nflats, iterations, blocksize]
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


def tEMFQ_GPU_(count, flat, angles, pad, interpolation, 
               det_pixel, tv_reg, iterations, gpus, blocksize, obj=None):
    """ Wrapper for MultiGPU/CUDA function of the 
    Transmission Expectation maximization (EM) method for 3D tomographic reconstructions in 
    parallel beam geometry. Flat (or empty) here is defined by a measurement without
    a sample, to measure the background.

    This EM method uses the Fourier Slice Theorem (FST) for the forward operator and 
    Backprojection Slice Theorem (BST) for the inverse operator.

    Args:
        counts (ndarray): Photon counts 3D data with shape (slices,angles,lenght) 
        flat (ndarray): Flat (or background) data with shape (slices,1,lenght) 
        angles (float list): List of angles in radians
        pad (int): Data padding - Integer multiple of the data size (0,1,2, etc...)
        interpolation (str): Type of interpolation. Options: \'nearest\' or \'bilinear\'
        det_pixel (float): Detector pixel size in meters
        tv_reg (float): Total variation regularization parameter
        iterations (int): EM iterations
        gpus (int list): List of gpus
        obj (ndarray,optional): Initial guess for the EM iterations with same shape as counts [default: zeros array]
        
    Returns:
        (ndarray): stacking 3D reconstructed volume (z,y,x) or 2D reconstructed sinograms (y,x)
    """
    if len(count.shape) == 2:
        nslices = 1
    else:
        nslices = count.shape[0]
    
    if len(flat.shape) == 2:
        nflats = 1
    else:
        nflats = flat.shape[0]

    nangles     = count.shape[-2]
    nrays       = count.shape[-1]

    objsize     = nrays
    
    ngpus       = len(gpus)
    gpus        = numpy.array(gpus)
    gpus        = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr    = gpus.ctypes.data_as(ctypes.c_void_p)

    # obj can be the initial guess
    if obj is not None:
        obj     = CNICE(obj)
        obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)
    else:
        obj     = numpy.ones([nslices, objsize, objsize], dtype=numpy.float32)
        obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    count       = CNICE(count) 
    count_ptr   = count.ctypes.data_as(ctypes.c_void_p) 

    flat        = CNICE(flat) 
    flat_ptr    = flat.ctypes.data_as(ctypes.c_void_p) 
    
    angles      = numpy.array(angles)
    angles      = CNICE(angles) 
    angles_ptr  = angles.ctypes.data_as(ctypes.c_void_p) 

    padx,pady,padz         = pad
    det_pixelx, det_pixely = det_pixel

    # padd = padx * nrays
    # logger.info(f'Set EM Frequency pad value as {padx} x horizontal dimension = ({padd}).')

    param_int     = [nrays, nangles, nslices, objsize, 
                     padx, pady, padz, nflats, iterations, interpolation, blocksize]
    param_int     = numpy.array(param_int)
    param_int     = CNICE(param_int,numpy.int32)
    param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

    param_float     = [det_pixelx, det_pixely, tv_reg]
    param_float     = numpy.array(param_float)
    param_float     = CNICE(param_float,numpy.float32)
    param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)


    libraft.get_tEM_FQ_MultiGPU( gpus_ptr, ctypes.c_int(ngpus),
                    count_ptr, obj_ptr, angles_ptr, flat_ptr,
                    param_float_ptr, param_int_ptr)

    return obj


