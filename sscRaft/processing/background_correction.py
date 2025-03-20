# Authors: Paola Ferraz, Giovanni L. Baraldi, Gilberto Martinez

from ..rafttypes import *
from ..io.io_ import *

def _background_correctionGPU(frames, flat, dark, gpus = [0], is_log = False, blocksize = 0):
    """ GPU function to correct tomography projections (or frames) background 
    with flat (or empty) and dark. Flat (or empty) here is defined by a measurement without
    a sample, to measure the background.
    
    Can be computed in two ways.

    .. math::
        T = - \log{ ( \\frac{D - D_d}{D_f - D_d} ) }

    for transmission tomography, and

    .. math::
        T = \\frac{D - D_d}{D_f - D_d}

    for phase contrast tomography. Where :math:`T` is the corrected tomogram, :math:`D` is the projection volume, 
    :math:`D_f` is the flat projections and :math:`D_d` is the dark projections

    Args:
        frames (ndarray): Frames (or projections) of size [slices, angles, lenght]
        flat   (ndarray): Flat of size [slices, number of flats, lenght]
        dark   (ndarray): Dark of size [slices, lenght]
        gpus  (int list, optional): List of GPUs [Default: [0]]
        is_log    (bool, optional): Apply ``- logarithm()`` or not [Default: False]
        blocksize  (int, optional): Block of slices size to be processed in one GPU. \'blocksize = 0\' computes it automatically considering the available GPU memory [Default: 0]

    Returns:
        (ndarray): Corrected frames (or projections) of dimension [slices, angles, lenght]

    * One or MultiGPUs. 
    """ 
        
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = frames.shape[-1]
    nangles  = frames.shape[-2]
    
    if is_log:
        is_log = 1
        logger.info(f'Returning corrected data with -log() applied.')
    else:
        is_log = 0
        logger.info(f'Returning corrected data without -log() applied.')

    if len(frames.shape) == 2:
        nslices = 1
    elif len(frames.shape) == 3:
        nslices = frames.shape[0]
    else:
        message_error = f'Incorrect data dimension: {frames.shape}! It accepts only 2- or 3-dimension array.'
        logger.error(message_error)
        raise ValueError(message_error)
            
    if len(flat.shape) == 2:
        nflats  = 1
    elif len(flat.shape) == 3:
        nflats = flat.shape[-2]
    else:
        message_error = f'Incorrect flat dimension: {flat.shape}! It accepts only 2- or 3-dimension array.'
        logger.error(message_error)
        raise ValueError(message_error)
    
    if len(dark.shape) == 2:
        pass
    elif len(dark.shape) == 3:
        dark = dark[:,0,:]
    else:
        message_error = f'Incorrect dark dimension: {dark.shape}! It accepts only 2- or 3-dimension array.'
        logger.error(message_error)
        raise ValueError(message_error)

    logger.info(f'Number of flats is {nflats}.')

    if flat.shape[-1] != nrays or flat.shape[0] != nslices:
        message_error = f'Flat dimension ({flat.shape[0]},{flat.shape[-1]}) does not match data dimension ({frames.shape[0]},{frames.shape[-1]}).'
        logger.error(message_error)
        raise ValueError(f'Flat dimension ({flat.shape[0]},{flat.shape[-1]}) does not match data dimension ({frames.shape[0]},{frames.shape[-1]}).')

    if dark.shape[-1] != nrays or dark.shape[0] != nslices:
        message_error = f'Dark dimension ({dark.shape[0]},{dark.shape[-1]}) does not match data dimension ({frames.shape[0]},{frames.shape[-1]}).'
        logger.error(message_error)
        raise ValueError(message_error)

    if nflats > 1:
        logger.info(f'Interpolating flats before and after.')

    logger.info(f'Flat dimension is ({flat.shape[0]},{nflats},{flat.shape[-1]}) = (slices,number of flats,rays).')
    logger.info(f'Dark dimension is ({dark.shape[0]},1,{dark.shape[-1]}) = (slices,number of darks,rays).')

    flat       = CNICE(flat)
    flat_ptr   = flat.ctypes.data_as(ctypes.c_void_p)

    dark       = CNICE(dark)
    dark_ptr   = dark.ctypes.data_as(ctypes.c_void_p)
    
    frames     = CNICE(frames)
    frames_ptr = frames.ctypes.data_as(ctypes.c_void_p)

    # print('BG frames_ptr: ', frames_ptr)

    libraft.getBackgroundCorrectionMultiGPU(gpus_ptr, ctypes.c_int(ngpus), 
            frames_ptr, flat_ptr, dark_ptr, 
            c_int(nrays), c_int(nangles), c_int(nslices), 
            c_int(nflats), c_int(is_log), c_int(blocksize))

    return frames 

def correct_projections(frames, flat, dark, dic, **kwargs):
    """ Function to correct tomography projections (or frames) with flat and dark. 
    Flat (or empty) here is defined by a measurement without a sample, to measure the background.
    
    Can be computed in two ways.

    .. math::
        T = - \log{ ( \\frac{D - D_d}{D_f - D_d} ) }

    for transmission tomography, and

    .. math::
        T = \\frac{D - D_d}{D_f - D_d}

    for phase contrast tomography. Where :math:`T` is the corrected tomogram, :math:`D` is the projection volume, 
    :math:`D_f` is the flat projections and :math:`D_d` is the dark projections

    Args:
        frames (ndarray): Frames (or projections) of size [angles, slices, lenght]
        flat   (ndarray): Flat of size [number of flats, slices, lenght]
        dark   (ndarray): Dark of size [slices, lenght]
        dic (dictionary): Dictionary with the parameters info.

    Returns:
        (ndarray): Corrected frames (or projections) of dimension [slices, angles, lenght]
    
    Dictionary parameters:
    
        * ``dic['gpu']`` (int list): List of GPUs [Default: [0]]
        * ``dic['uselog']`` (bool, optional): Apply ``- logarithm()`` or not [Default: False]
        * ``dic['blocksize']`` (int, optional): Block of slices size to be processed in one GPU. \'blocksize = 0\' computes it automatically considering the available GPU memory [Default: 0]

    * One or MultiGPUs. 
    * Calls function ``_background_correctionGPU()``.
    """        
    
    # Set dictionary parameters:
    required  = ('gpu',)
    optional  = ('uselog','blocksize')
    defaut    = (False,0)
    
    dic = SetDictionary(dic,required,optional,defaut)

    gpus      = dic['gpu']
    is_log    = dic['uselog']
    blocksize = dic['blocksize']
    
    frames    = numpy.swapaxes(frames,0,1)
    flat      = numpy.swapaxes(flat,0,1)
    dark      = numpy.swapaxes(dark,0,1)

    frames    = _background_correctionGPU( frames, flat, dark, gpus, is_log, blocksize ) 

    return frames

def correct_background(frames, flat, dark, gpus = [0], is_log = False, blocksize = 0):
    """ Function to correct tomography projections (or frames) background 
    with flat (or empty) and dark. Flat (or empty). Flat is a measurement without
    a sample, to measure the background. Dark is a measurements without a beam and sample, 
    to measure detector pixel response.
    
    Can be computed in two ways.

    .. math::
        T = - \log{ ( \\frac{D - D_d}{D_f - D_d} ) }

    for transmission tomography, and

    .. math::
        T = \\frac{D - D_d}{D_f - D_d}

    for phase contrast tomography. Where :math:`T` is the corrected tomogram, :math:`D` is the measurements volume, 
    :math:`D_f` is the flat measurement and :math:`D_d` is the dark measurement.

    Args:
        frames (ndarray): Frames (or projections) of size [slices, angles, lenght]
        flat   (ndarray): Flat of size [slices, number of flats, lenght]
        dark   (ndarray): Dark of size [slices, lenght]
        gpus  (int list, optional): List of GPUs [Default: [0]].
        is_log (bool, optional): Apply ``- logarithm()`` or not [Default: False]
        blocksize (int, optional): Block of slices size to be processed in one GPU. \'blocksize = 0\' computes it automatically considering the available GPU memory [Default: 0]


    Returns:
        (ndarray): Corrected frames (or projections) of dimension [slices, angles, lenght].

    * One or MultiGPUs. 
    * Calls function ``_background_correctionGPU()``.
    """        
    
    frames = _background_correctionGPU( frames, flat, dark, gpus, is_log, blocksize ) 

    return frames
