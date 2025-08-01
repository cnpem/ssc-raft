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
        blocksize (int): Number of simultaneous reconstructed slices 
        obj (ndarray, optional): Reconstructed 3D object array and initial guess [default: zeros array]
        
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
    else:
        obj     = numpy.ones([nslices, objsize, objsize], dtype=numpy.float32)
        obj     = CNICE(obj)
    obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    tomo          = CNICE(tomo) #sino pointer
    tomo_ptr      = tomo.ctypes.data_as(ctypes.c_void_p) 
    
    angles        = numpy.array(angles)
    angles        = CNICE(angles) #angles pointer
    angles_ptr    = angles.ctypes.data_as(ctypes.c_void_p) 

    padding       = 0
    pixel         = 1.0
    z2            = 1.0
    energy        = 1.0
    wavelength    = 1.0

    tomo_dim      = dimension((  nrays, nangles, nslices), (padding,       0, 0), blocksize = blocksize)
    obj_dim       = dimension((objsize, objsize, nslices), (padding, padding, 0), blocksize = blocksize)
    
    geometry      = define_geometry(detector_pixel = (pixel, pixel),
                                    obj_pixel      = (pixel, pixel),
                                    z1             = (0,0),
                                    z2             = (z2,z2),
                                    magnitude      = (1.0,1.0), 
                                    energy         = energy, 
                                    wavelength     = wavelength)

    ReconParam   = REC(method               = 0, 
                       filter               = 0, 
                       filter_reg           = 1.0,  
                       paganin_slices       = 0.0, 
                       iterations           = iterations, 
                       rotation_axis_offset = 0,
                       total_variation      = 0, 
                       interpolation        = 0)

    libraft.get_eEM_RT_MultiGPU(tomo_dim, obj_dim, geometry, ReconParam, 
                                gpus_ptr, ctypes.c_int(ngpus),
                                obj_ptr, tomo_ptr, angles_ptr)

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
        blocksize (int): Number of simultaneous reconstructed slices 
        obj (ndarray, optional): Reconstructed 3D object array and initial guess [default: zeros array]
        
    Returns:
        (ndarray): stacking 3D reconstructed volume (z,y,x) or 2D reconstructed sinograms (y,x)
    """
    # MultiGPU withou semafaros

    if len(counts.shape) == 2:
        nslices = 1
    else:
        nslices = counts.shape[0]

    if len(flat.shape) == 2:
        nflats = 1
    else:
        nflats = flat.shape[0]

    nangles       = counts.shape[-2]
    nrays         = counts.shape[-1]

    objsize       = counts.shape[-1]

    counts       *= -1
    counts        = numpy.exp(counts, counts)
    
    ngpus         = len(gpus)
    gpus          = numpy.array(gpus)
    gpus          = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr      = gpus.ctypes.data_as(ctypes.c_void_p)

    if obj is not None:
        obj     = CNICE(obj)
    else:
        obj     = numpy.ones([nslices, objsize, objsize], dtype=numpy.float32)
        obj     = CNICE(obj)
    obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    counts        = CNICE(counts) 
    counts_ptr    = counts.ctypes.data_as(ctypes.c_void_p) 

    flat          = CNICE(flat) 
    flat_ptr      = flat.ctypes.data_as(ctypes.c_void_p) 
    
    angles        = numpy.array(angles)
    angles        = CNICE(angles) #angles pointer
    angles_ptr    = angles.ctypes.data_as(ctypes.c_void_p) 

    padding       = 0
    pixel         = 1.0
    z2            = 1.0
    energy        = 1.0
    wavelength    = 1.0

    tomo_dim      = dimension((  nrays, nangles, nslices), (padding,       0, 0), blocksize = blocksize)
    obj_dim       = dimension((objsize, objsize, nslices), (padding, padding, 0), blocksize = blocksize)
    
    geometry      = define_geometry(detector_pixel = (pixel, pixel),
                                    obj_pixel      = (pixel, pixel),
                                    z1             = (0,0),
                                    z2             = (z2,z2),
                                    magnitude      = (1.0,1.0), 
                                    energy         = energy, 
                                    wavelength     = wavelength)

    ReconParam   = REC(method               = 0, 
                       filter               = 0, 
                       filter_reg           = 1.0,  
                       paganin_slices       = 0.0, 
                       iterations           = iterations, 
                       rotation_axis_offset = 0,
                       total_variation      = 0, 
                       interpolation        = 0)

    libraft.get_tEM_RT_MultiGPU(tomo_dim, obj_dim, geometry, ReconParam,
                                gpus_ptr, ctypes.c_int(ngpus),
                                obj_ptr, counts_ptr, flat_ptr, angles_ptr)

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
    else:
        obj     = numpy.ones([nslices, objsize, objsize], dtype=numpy.float32)
        obj     = CNICE(obj)
    obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    count       = CNICE(count) 
    count_ptr   = count.ctypes.data_as(ctypes.c_void_p) 

    flat        = CNICE(flat) 
    flat_ptr    = flat.ctypes.data_as(ctypes.c_void_p) 
    
    angles      = numpy.array(angles)
    angles      = CNICE(angles) 
    angles_ptr  = angles.ctypes.data_as(ctypes.c_void_p) 


    interp      = setInterpolation(interpolation)
    padding     = pad
    pixel       = det_pixel

    z2          = 1.0
    energy      = 1.0
    wavelength  = 1.0

    tomo_dim    = dimension((  nrays, nangles, nslices), (padding,       0, 0), blocksize = blocksize)
    obj_dim     = dimension((objsize, objsize, nslices), (padding, padding, 0), blocksize = blocksize)
    
    geometry    = define_geometry(detector_pixel = (pixel, pixel),
                                  obj_pixel      = (pixel, pixel),
                                  z1             = (0,0),
                                  z2             = (z2,z2),
                                  magnitude      = (1.0,1.0), 
                                  energy         = energy, 
                                  wavelength     = wavelength)

    ReconParam  = REC(method               = 0, 
                      filter               = 0, 
                      filter_reg           = 1.0,  
                      paganin_slices       = 0.0, 
                      iterations           = iterations, 
                      rotation_axis_offset = 0,
                      total_variation      = tv_reg, 
                      interpolation        = interp)

    libraft.get_tEM_FQ_MultiGPU(tomo_dim, obj_dim, geometry, ReconParam, 
                                gpus_ptr, ctypes.c_int(ngpus),
                                count_ptr, obj_ptr, angles_ptr, flat_ptr)

    return obj


