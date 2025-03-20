from ...rafttypes import *
from ...io.io_ import *

def TitarenkoRingsGPU(tomogram, gpus, rings_lambda, rings_block, blocksize = 0):
    """ Rings artifacts reduction by Generalized Titarenko\'s algorithm [1]_.  

    Args:
        tomogram (ndarray): Tomogram (3D) or Sinogram (2D). The axes are [slices, angles, lenght] (3D) or [angles, lenght] (2D).
        gpus (int list): List of GPUs
        rings_lambda (float): Titarenko\'s regularization value. Values between [0,1]. The value -1 compute this parameter automatically
        rings_block (int): Blocks of sinograms to be used. Even values between [1,20]
        blocksize (int, optional): Block of slices size to be processed in one GPU. \'blocksize = 0\' computes it automatically considering the available GPU memory [Default: 0]


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

    tomogram            = CNICE(tomogram)
    tomogram_ptr         = tomogram.ctypes.data_as(ctypes.c_void_p)

    libraft.getTitarenkoRingsMultiGPU(gpusptr, ctypes.c_int(ngpus), tomogram_ptr, 
            ctypes.c_int(nrays), ctypes.c_int(nangles), ctypes.c_int(nslices), 
            ctypes.c_float(rings_lambda), ctypes.c_int(rings_block),
            ctypes.c_int(blocksize))

    return tomogram

