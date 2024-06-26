from ...rafttypes import *
from ...processing.io import *

def TitarenkoRingsGPU(tomogram, gpus, rings_lambda, rings_block):
    """ Rings artifacts reduction by Generalized Titarenko\'s algorithm.  

    Args:
        tomogram (ndarray): Tomogram (3D) or Sinogram (2D). The axes are [slices, angles, lenght] (3D) or [angles, lenght] (2D).
        gpus (int list): List of GPUs
        rings_lambda (float): Titarenko\'s regularization value. Values between [0,1]. The value -1 compute this parameter automatically
        rings_block (int): Blocks of sinograms to be used. Even values between [1,20]

    Returns:
        (ndarray): Tomogram (3D) or Sinogram (2D). The axes are [slices, angles, lenght] (3D) or [angles, lenght] (2D).

    References:

        .. [1] Miqueles, E.X., Rinkel, J., O'Dowd, F. and Bermudez, J.S.V. (2014). Generalized Titarenko\'s algorithm for ring artefacts reduction. J. Synchrotron Rad, 21, 1333-1346. DOI: https://doi.org/10.1107/S1600577514016919
    
    """
    ngpus = len(gpus)

    gpus = numpy.array(gpus)
    gpus = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpusptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays   = tomogram.shape[-1]
    nangles = tomogram.shape[-2]

    if len(tomogram.shape) == 2:
        nslices = 1
    elif len(tomogram.shape) == 3:
        nslices = tomogram.shape[0]
    else:
        message_error = f'Incorrect tomogram dimension:{tomogram.shape}! It accepts 2D or 3D arrays only.'
        logger.error(message_error)
        raise ValueError(message_error)

    if rings_lambda == 0:
        logger.warning(f'No Titarenko\'s regularization: set to {rings_lambda}.')
        return tomogram
    elif rings_lambda < 0:
        logger.info(f'Computing automatic Titarenko\'s regularization parameter.')
    else:
        logger.info(f'Titarenko\'s regularization set to {rings_lambda}.')  

    tomogram            = numpy.ascontiguousarray(tomogram.astype(numpy.float32))
    tomogram_ptr         = tomogram.ctypes.data_as(ctypes.c_void_p)

    libraft.getTitarenkoRingsMultiGPU(gpusptr, ctypes.c_int(ngpus), tomogram_ptr, 
            ctypes.c_int(nrays), ctypes.c_int(nangles), ctypes.c_int(nslices), 
            ctypes.c_float(rings_lambda), ctypes.c_int(rings_block))

    return tomogram

def rings(tomogram, dic, **kwargs):
    """Apply rings correction on tomogram.

    Args:
        tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, lenght] (3D) or [angles, lenght] (2D).
        dic   (dictionary): Dictionary with the parameters.

    Returns:
        (ndarray): Tomogram (3D) or Sinogram (2D). The axes are [slices, angles, lenght] (3D) or [angles, lenght] (2D).
    
    * One or MultiGPUs
    * Calls function ``TitarenkoRingsGPU()``

    Dictionary parameters:

        * ``dic['gpu']`` (int list): List of GPUs to use [required]
        * ``dic['lambda rings']`` (float,optional): Regularization parameter. Values between [0,1] [default: -1 (automatic computation)]
        * ``dic['rings block']`` (int,optional): Blocks of sinograms to be used. Even values between [1,20] [default: 1]   
    """
    required = ('gpu',)
    optional = ('lambda rings','rings block')
    default  = (-1,1)

    dic = SetDictionary(dic,required,optional,default)

    gpus = dic['gpu']

    rings_lambda = dic['lambda rings']
    rings_block  = dic['rings block']

    tomogram = TitarenkoRingsGPU( tomogram, gpus, rings_lambda, rings_block ) 

    return tomogram