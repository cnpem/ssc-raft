# Authors: Giovanni L. Baraldi, Gilberto Martinez
from ...rafttypes import *

def fbpGPU(tomogram, angles, gpus, dic, obj=None):
    """Wrapper fo MultiGPU/CUDA function that computes the reconstruction of a parallel beam 
    tomogram using the Filtered Backprojection (FBP) method.

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        angles (float list): List of angles in radians
        gpus (int list): List of gpus
        dic (dict): Dictionary with parameters info
        obj (ndarray, optional): Reconstructed 3D object array [default: None]

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

    Dictionary parameters:

        * ``dic['filter']`` (str): Filter type [required]

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')
        
        * ``dic['detectorPixel[m]']`` (float,optional): Detector pixel size in meters [Default: 1.0]
        * ``dic['beta/delta']`` (float,optional): Paganin by slices method ``beta/delta`` ratio [Default: 0.0 (no Paganin applied)]
        * ``dic['z2[m]']`` (float,optional): Sample-Detector distance in meters used on Paganin by slices method. [Default: 1.0]
        * ``dic['energy[eV]']`` (float,optional): beam energy in eV used on Paganin by slices method. [Default: 1.0 ]
        * ``dic['regularization']`` (float,optional): Regularization value for filter ( value >= 0 ) [Default: 1.0]
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [Default: 2]
        * ``dic['blocksize']`` (int,optional): Block of slices to be simulteneously computed [Default: 0 (automatically)]
        * ``dic['rotation axis offset']`` (float,optional): Rotation axis deviation value [Default: 0.0]

    """        
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = tomogram.shape[-1]
    nangles  = tomogram.shape[-2]
    
    if len(tomogram.shape) == 2:
        nslices = 1
    else:
        nslices = tomogram.shape[0]
    
    objsize        = nrays

    filter_type    = FilterNumber(dic['filter'])
    beta_delta     = dic['beta/delta']
    regularization = dic['regularization']
    offset         = dic['rotation axis offset']
    blocksize      = dic['blocksize']
    energy         = dic['energy[eV]']
    z2             = dic['z2[m]']
    pixelx, pixely = dic['detectorPixel[m]'],dic['detectorPixel[m]']

    if beta_delta != 0.0:
        beta_delta = 1.0 / beta_delta
    else:
        beta_delta     = 0.0
        z2             = 0.0
        energy         = 1.0

    padx, pady, padz  = dic['padding'],0,0 # (padx, pady, padz)

    pad = (padx) * nrays
    logger.info(f'Set FBP RT pad value as {padx} x horizontal dimension = ({pad}).')
    aux = objsize + pad
    logger.info(f'Ob size {aux}.')


    tomogram     = CNICE(tomogram) 
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    if obj is None:
        obj      = numpy.zeros([nslices, objsize + pad, objsize + pad], dtype=numpy.float32)
        obj      = CNICE(obj)
    obj_ptr      = obj.ctypes.data_as(ctypes.c_void_p)

    angles       = numpy.array(angles)
    angles       = CNICE(angles) 
    angles_ptr   = angles.ctypes.data_as(ctypes.c_void_p) 

    param_int     = [nrays, nangles, nslices, objsize, 
                     padx, pady, padz, filter_type, 0, blocksize]
    param_int     = numpy.array(param_int)
    param_int     = CNICE(param_int,numpy.int32)
    param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

    param_float     = [beta_delta, regularization, energy, z2, pixelx, pixely, offset]
    param_float     = numpy.array(param_float)
    param_float     = CNICE(param_float,numpy.float32)
    param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)


    # bShiftCenter = dic['shift center']

    libraft.getFBPMultiGPU(gpus_ptr, ctypes.c_int(ngpus), 
        obj_ptr, tomogram_ptr, angles_ptr, 
        param_float_ptr, param_int_ptr)

    return obj

def bstGPU(tomogram, angles, gpus, dic, obj = None):
    """Wrapper fo MultiGPU/CUDA function that computes the reconstruction of a parallel beam 
    tomogram using the Backprojection Slice Theorem (BST) method.

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        angles (float list): List of angles in radians
        gpus (int list): List of gpus
        dic (dict): Dictionary with parameters info
        obj (ndarray, optional): Reconstructed 3D object array [default: None]

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

    Dictionary parameters:

        * ``dic['filter']`` (str): Filter type [required]

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')
        
        * ``dic['detectorPixel[m]']`` (float,optional): Detector pixel size in meters [Default: 1.0]
        * ``dic['beta/delta']`` (float,optional): Paganin by slices method ``beta/delta`` ratio [Default: 0.0 (no Paganin applied)]
        * ``dic['z2[m]']`` (float,optional): Sample-Detector distance in meters used on Paganin by slices method. [Default: 1.0]
        * ``dic['energy[eV]']`` (float,optional): beam energy in eV used on Paganin by slices method. [Default: 1.0]
        * ``dic['regularization']`` (float,optional): Regularization value for filter ( value >= 0 ) [Default: 1.0]
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [Default: 2]
        * ``dic['blocksize']`` (int,optional): Block of slices to be simulteneously computed [Default: 0 (automatically)]
        * ``dic['rotation axis offset']`` (float,optional): Rotation axis deviation value [Default: 0.0]

    References:

        .. [1] Miqueles, X. E. and Koshev, N. and Helou, E. S. (2018). A Backprojection Slice Theorem for Tomographic Reconstruction. IEEE Transactions on Image Processing, 27(2), p. 894-906. DOI: https://doi.org/10.1109/TIP.2017.2766785.
    
    """         
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = tomogram.shape[-1]
    nangles  = tomogram.shape[-2]
    
    if len(tomogram.shape) == 2:
            nslices = 1
    else:
            nslices = tomogram.shape[0]
    
    objsize        = nrays

    filter_type    = FilterNumber(dic['filter'])
    beta_delta     = dic['beta/delta']
    regularization = dic['regularization']
    offset         = int(dic['rotation axis offset'])
    blocksize      = dic['blocksize']
    energy         = dic['energy[eV]']
    z2             = dic['z2[m]']
    pixelx, pixely = dic['detectorPixel[m]'],dic['detectorPixel[m]']

    if beta_delta != 0.0:
        beta_delta = 1.0 / beta_delta
    else:
        beta_delta     = 0.0
        z2             = 0.0
        energy         = 1.0
        
    padx, pady, padz  = dic['padding'] + 2,0,0 # (padx, pady, padz)

    pad = (padx + 2) * nrays
    logger.info(f'Set BST pad value as {padx} x horizontal dimension = ({pad}).')

    tomogram     = CNICE(tomogram) 
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    if obj is None:
        obj      = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
        obj      = CNICE(obj)
    obj_ptr      = obj.ctypes.data_as(ctypes.c_void_p)

    angles       = numpy.array(angles)
    angles       = CNICE(angles) 
    angles_ptr   = angles.ctypes.data_as(ctypes.c_void_p) 

    param_int     = [nrays, nangles, nslices, objsize, 
                     padx, pady, padz, filter_type, 0, blocksize]
    param_int     = numpy.array(param_int)
    param_int     = CNICE(param_int,numpy.int32)
    param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

    param_float     = [beta_delta, regularization, energy, z2, pixelx, pixely, offset]
    param_float     = numpy.array(param_float)
    param_float     = CNICE(param_float,numpy.float32)
    param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)


    libraft.getBSTMultiGPU(gpus_ptr, ctypes.c_int(ngpus), 
        obj_ptr, tomogram_ptr, angles_ptr, 
        param_float_ptr, param_int_ptr)

    return obj


