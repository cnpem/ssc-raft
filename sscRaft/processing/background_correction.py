# Authors: Paola Ferraz, Giovanni L. Baraldi, Gilberto Martinez

from ..rafttypes import *
from ..io.io_ import *

def correct_background(data, flat, dark, gpus = [0], is_log = True, blocksize = 0, nstreams = 1):
    """ GPU function to correct tomography data background 
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
        data (ndarray): Measurament data with axis (slices, angles, lenght) 
        flat   (ndarray): Flat with size (slices, number of flats, lenght)
        dark   (ndarray): Dark with size (slices, lenght)
        gpus  (int list, optional): List of GPUs [Default: [0]]
        is_log    (bool, optional): Apply ``- logarithm()`` or not [Default: True]
        blocksize  (int, optional): Block of slices size to be processed in one GPU. ``blocksize = 0`` computes it automatically considering the available GPU memory [Default: 0]
        nstreams (int, optional): Number of cuda streams `` > 0`` [Default: 1]

    Returns:
        (ndarray): Corrected tomogram with axis (slices, angles, lenght).

    * One or MultiGPUs. 
    """ 
        
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    sizex    = data.shape[-1]
    sizey    = data.shape[-2]

    order_axis = set_input_axis_order('slices_angles_lenght')
    
    if is_log:
        logger.info(f'Returning corrected data with \'-log()\' applied.')
    else:
        logger.info(f'No \'-log()\' applied.')

    if len(data.shape) == 2:
        sizez = 1
    elif len(data.shape) == 3:
        sizez = data.shape[0]
    else:
        message_error = f'Incorrect data dimension: {data.shape}! It accepts only 2- or 3-dimension array.'
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

    if nflats > 1:
        logger.info(f'Interpolating flats before and after.')

    flat       = CNICE(flat)
    flat_ptr   = flat.ctypes.data_as(ctypes.c_void_p)

    dark       = CNICE(dark)
    dark_ptr   = dark.ctypes.data_as(ctypes.c_void_p)
    
    data     = CNICE(data)
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)

    libraft.getBackgroundCorrectionMultiGPU(gpus_ptr, ctypes.c_int(ngpus), 
            data_ptr, flat_ptr, dark_ptr, 
            ctypes.c_int(sizex), ctypes.c_int(sizey), ctypes.c_int(sizez), 
            ctypes.c_int(nflats), ctypes.c_int(is_log), ctypes.c_int(order_axis), 
            ctypes.c_int(blocksize), ctypes.c_int(nstreams))

    return data 

def correct_background_frames(data, flat, dark, gpus = [0], is_log = True, blocksize = 0, nstreams = 1):
    """ GPU function to correct tomography data background 
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
        data (ndarray): Measurament data with axis (angles, slices, lenght) 
        flat   (ndarray): Flat with size (number of flats, slices, lenght)
        dark   (ndarray): Dark with size (slices, lenght)
        gpus  (int list, optional): List of GPUs [Default: [0]]
        is_log    (bool, optional): Apply ``- logarithm()`` or not [Default: False]
        blocksize  (int, optional): Block of slices size to be processed in one GPU. ``blocksize = 0`` computes it automatically considering the available GPU memory [Default: 0]
        nstreams (int, optional): Number of cuda streams `` > 0`` [Default: 1]

    Returns:
        (ndarray): Corrected tomogram with axis (angles, slices, lenght).

    * One or MultiGPUs. 
    """ 
        
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    sizex    = data.shape[-1]
    sizey    = data.shape[-2]

    order_axis = set_input_axis_order('angles_slices_lenght')
    
    if is_log:
        logger.info(f'Returning corrected data with \'-log()\' applied.')
    else:
        logger.info(f'No \'-log()\' applied.')

    if len(data.shape) == 2:
        sizez = 1
    elif len(data.shape) == 3:
        sizez = data.shape[0]
    else:
        message_error = f'Incorrect data dimension: {data.shape}! It accepts only 2- or 3-dimension array.'
        logger.error(message_error)
        raise ValueError(message_error)
            
    if len(flat.shape) == 2:
        nflats  = 1
    elif len(flat.shape) == 3:
        nflats = flat.shape[0]
    else:
        message_error = f'Incorrect flat dimension: {flat.shape}! It accepts only 2- or 3-dimension array.'
        logger.error(message_error)
        raise ValueError(message_error)
    
    if len(dark.shape) == 2:
        pass
    elif len(dark.shape) == 3:
        dark = dark[0]
    else:
        message_error = f'Incorrect dark dimension: {dark.shape}! It accepts only 2- or 3-dimension array.'
        logger.error(message_error)
        raise ValueError(message_error)

    logger.info(f'Number of flats is {nflats}.')

    if nflats > 1:
        logger.info(f'Interpolating flats before and after.')

    flat       = CNICE(flat)
    flat_ptr   = flat.ctypes.data_as(ctypes.c_void_p)

    dark       = CNICE(dark)
    dark_ptr   = dark.ctypes.data_as(ctypes.c_void_p)
    
    data     = CNICE(data)
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)

    libraft.getBackgroundCorrectionMultiGPU(gpus_ptr, ctypes.c_int(ngpus), 
            data_ptr, flat_ptr, dark_ptr, 
            ctypes.c_int(sizex), ctypes.c_int(sizey), ctypes.c_int(sizez), 
            ctypes.c_int(nflats), ctypes.c_int(is_log), ctypes.c_int(order_axis), 
            ctypes.c_int(blocksize), ctypes.c_int(nstreams))

    return data 

# def correct_background_transpose(data, flat, dark, gpus = [0], is_log = True, blocksize = 0, nstreams = 1):
#     """ GPU function to correct tomography data background 
#     with flat (or empty) and dark. Flat (or empty) here is defined by a measurement without
#     a sample, to measure the background.
    
#     Can be computed in two ways.

#     .. math::
#         T = - \log{ ( \\frac{D - D_d}{D_f - D_d} ) }

#     for transmission tomography, and

#     .. math::
#         T = \\frac{D - D_d}{D_f - D_d}

#     for phase contrast tomography. Where :math:`T` is the corrected tomogram, :math:`D` is the projection volume, 
#     :math:`D_f` is the flat projections and :math:`D_d` is the dark projections

#     Args:
#         data (ndarray): Measurament data with axis (angles, slices, lenght)
#         flat   (ndarray): Flat with size (number of flats, slices, lenght)
#         dark   (ndarray): Dark with size (slices, lenght)
#         gpus  (int list, optional): List of GPUs [Default: [0]]
#         is_log    (bool, optional): Apply ``- logarithm()`` or not [Default: False]
#         blocksize  (int, optional): Block of slices size to be processed in one GPU. ``blocksize = 0`` computes it automatically considering the available GPU memory [Default: 0]
#         nstreams (int, optional): Number of cuda streams `` > 0`` [Default: 1]
#     Returns:
#         (ndarray): Corrected tomogram with axis (slices, angles, lenght).

#     * One or MultiGPUs. 
#     """ 
        
#     ngpus    = len(gpus)
#     gpus     = numpy.array(gpus)
#     gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
#     gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

#     sizex    = data.shape[-1]
#     sizey    = data.shape[-2]

#     order_axis = set_input_axis_order('transpose_lenght')
    
#     if is_log:
#         logger.info(f'Returning corrected data with \'-log()\' applied.')
#     else:
#         logger.info(f'No \'-log()\' applied.')

#     if len(data.shape) == 2:
#         sizez = 1
#     elif len(data.shape) == 3:
#         sizez = data.shape[0]
#     else:
#         message_error = f'Incorrect data dimension: {data.shape}! It accepts only 2- or 3-dimension array.'
#         logger.error(message_error)
#         raise ValueError(message_error)
            
#     if len(flat.shape) == 2:
#         nflats  = 1
#     elif len(flat.shape) == 3:
#         nflats = flat.shape[0]
#     else:
#         message_error = f'Incorrect flat dimension: {flat.shape}! It accepts only 2- or 3-dimension array.'
#         logger.error(message_error)
#         raise ValueError(message_error)
    
#     if len(dark.shape) == 2:
#         pass
#     elif len(dark.shape) == 3:
#         dark = dark[0]
#     else:
#         message_error = f'Incorrect dark dimension: {dark.shape}! It accepts only 2- or 3-dimension array.'
#         logger.error(message_error)
#         raise ValueError(message_error)

#     logger.info(f'Number of flats is {nflats}.')

#     if nflats > 1:
#         logger.info(f'Interpolating flats before and after.')

#     flat       = CNICE(flat)
#     flat_ptr   = flat.ctypes.data_as(ctypes.c_void_p)

#     dark       = CNICE(dark)
#     dark_ptr   = dark.ctypes.data_as(ctypes.c_void_p)
    
#     data     = CNICE(data)
#     data_ptr = data.ctypes.data_as(ctypes.c_void_p)

#     libraft.getBackgroundCorrectionMultiGPU(gpus_ptr, ctypes.c_int(ngpus), 
#             data_ptr, flat_ptr, dark_ptr, 
#             ctypes.c_int(sizex), ctypes.c_int(sizey), ctypes.c_int(sizez), 
#             ctypes.c_int(nflats), ctypes.c_int(is_log), ctypes.c_int(order_axis), 
#             ctypes.c_int(blocksize), ctypes.c_int(nstreams))
    
#     new_shape = (sizey, sizez, sizex)
#     data = data.reshape(new_shape)

#     return data 