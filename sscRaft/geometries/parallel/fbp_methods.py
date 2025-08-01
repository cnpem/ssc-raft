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
        * ``dic['padding']`` (float,optional): Data padding - percentage of data size (0.1,0.5,1.0, etc...) [default: 0.0]
        * ``dic['padd_mode']`` (str,optional): Data padding mode - options: \'none\', \'zero\', \'ones\', \'edge\' [default: \'edge\'] 
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

    filter_type    = FilterNumber(dic.get('filter', 'ramp'))
    beta_delta     = dic.get('beta/delta', 0.0)
    regularization = dic.get('regularization', 1.0)
    offset         = dic.get('rotation axis offset', 0)
    blocksize      = dic.get('blocksize', 0)
    energy         = dic.get('energy[eV]', 1.0)
    z2             = dic.get('z2[m]', 1.0)
    pixel          = dic.get('detectorPixel[m]', 1.0)
    wavelength     = CONST/energy 
    padding        = dic.get('padding', 0.0)*100 # Multiply by 100 to get an integer value
    padd_mode      = PaddMode(dic.get('padd_mode', 'edge'))

    if beta_delta != 0.0:
        beta_delta = 1.0 / beta_delta
        paganin_slices_regularization = wavelength * z2 * numpy.pi * beta_delta / (pixel * pixel); 

    else:
        paganin_slices_regularization = 0.0

    # Object (reconstruction)
    objsize = nrays
    logger.info(f'Object size: (nslices, ny, nx) = ({nslices},{objsize},{objsize}).')
    
    tomogram     = CNICE(tomogram) 
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    if obj is None:
        obj      = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
        obj      = CNICE(obj)
    obj_ptr      = obj.ctypes.data_as(ctypes.c_void_p)

    angles       = numpy.array(angles)
    angles       = CNICE(angles) 
    angles_ptr   = angles.ctypes.data_as(ctypes.c_void_p) 

    tomo_dim     = dimension((  nrays, nangles, nslices), (padding,       0, 0), blocksize = blocksize, padd_mode = padd_mode)
    obj_dim      = dimension((objsize, objsize, nslices), (padding, padding, 0), blocksize = blocksize, padd_mode = padd_mode)
    
    geometry     = define_geometry(detector_pixel = (pixel, pixel),
                                   obj_pixel      = (pixel, pixel),
                                   z1             = (0,0),
                                   z2             = (z2,z2),
                                   magnitude      = (1.0,1.0), 
                                   energy         = energy, 
                                   wavelength     = wavelength)

    ReconParam   = REC(method               = 0, 
                       filter               = filter_type, 
                       filter_reg           = regularization,  
                       paganin_slices       = paganin_slices_regularization, 
                       iterations           = 0, 
                       rotation_axis_offset = offset,
                       total_variation      = 0, 
                       interpolation        = 0)
    
    libraft.getFBPMultiGPU(tomo_dim, obj_dim, geometry, ReconParam, 
                           gpus_ptr,ctypes.c_int(ngpus),
                           obj_ptr, tomogram_ptr, angles_ptr)
    
    ''' Correction scale (angular correction) for 
        cases where there are more than 180 degrees.
        Specially for testes with Mogno conebeam data 
        that is acquired in 360 degrees rotation.
    '''
    angles_range = numpy.abs(angles[-1] - angles[0])
    last_angle   = max( numpy.abs( angles[-1] ), numpy.abs( angles[0] ) )
    scale        = numpy.pi / last_angle
    
    if angles_range > numpy.pi:
        obj *= scale

    return obj

def bstGPU(tomogram, angles, gpus, dic, obj = None, nstreams = 0):
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
        * ``dic['padding']`` (float,optional): Data padding - percentage of data size (0.1,0.5,1.0, etc...) [default: 0.0]
        * ``dic['padd_mode']`` (str,optional): Data padding mode - options: \'none\', \'zero\', \'ones\', \'edge\' [default: \'edge\'] 
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

    filter_type    = FilterNumber(dic.get('filter', 'ramp'))
    beta_delta     = dic.get('beta/delta', 0.0)
    regularization = dic.get('regularization', 1.0)
    offset         = dic.get('rotation axis offset', 0)
    blocksize      = dic.get('blocksize', 0)
    energy         = dic.get('energy[eV]', 1.0)
    z2             = dic.get('z2[m]', 1.0)
    pixel          = dic.get('detectorPixel[m]', 1.0)
    wavelength     = CONST/energy 
    padding        = dic.get('padding', 0.0)*100 # Multiply by 100 to get an integer value
    padd_mode      = PaddMode(dic.get('padd_mode', 'edge'))

    if beta_delta != 0.0:
        beta_delta = 1.0 / beta_delta
        paganin_slices_regularization = wavelength * z2 * numpy.pi * beta_delta / (pixel * pixel); 
    else:
        paganin_slices_regularization = 0.0
        
    # Object (reconstruction)
    objsize = nrays
    logger.info(f'Object size: (nslices, ny, nx) = ({nslices},{objsize},{objsize}).')

    tomogram     = CNICE(tomogram) 
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    if obj is None:
        obj = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
        obj = CNICE(obj)
    obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    angles       = numpy.array(angles)
    angles       = CNICE(angles) 
    angles_ptr   = angles.ctypes.data_as(ctypes.c_void_p) 

    tomo_dim     = dimension((  nrays, nangles, nslices), (padding,       0, 0), blocksize = blocksize, padd_mode = padd_mode)
    obj_dim      = dimension((objsize, objsize, nslices), (padding, padding, 0), blocksize = blocksize, padd_mode = padd_mode)
    
    geometry     = define_geometry(detector_pixel = (pixel, pixel),
                                   obj_pixel      = (pixel, pixel),
                                   z1             = (0,0),
                                   z2             = (z2,z2),
                                   magnitude      = (1.0,1.0), 
                                   energy         = energy, 
                                   wavelength     = wavelength)

    ReconParam   = REC(method               = 0, 
                       filter               = filter_type, 
                       filter_reg           = regularization,  
                       paganin_slices       = paganin_slices_regularization, 
                       iterations           = 0, 
                       rotation_axis_offset = offset,
                       total_variation      = 0, 
                       interpolation        = 0)

    libraft.getBSTMultiGPU(tomo_dim, obj_dim, geometry, ReconParam,
                           gpus_ptr, ctypes.c_int(ngpus), 
                           obj_ptr, tomogram_ptr, angles_ptr, 
                           ctypes.c_int(nstreams))
    return obj


