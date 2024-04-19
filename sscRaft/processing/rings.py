from ..rafttypes import *
from .io import *
from .ringsmethods.titarenko import * 

def rings(tomogram, dic, **kwargs):
    """Apply rings correction on tomogram.

    Args:
        tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, lenght] (3D) or [angles, lenght] (2D).
        dic   (dictionary): Dictionary with the parameters.

    Returns:
        (ndarray): Tomogram (3D) or Sinogram (2D). The axes are [slices, angles, lenght] (3D) or [angles, lenght] (2D).
    
    * One or MultiGPUs
    * Calls function ``TitarenkoRingsGPU()`` [1]_

    Dictionary parameters:

        * ``dic['gpu']`` (int list): List of GPUs to use [required]
        * ``dic['lambda rings']`` (float,optional): Regularization parameter. Values between [0,1] [default: -1 (automatic computation)]
        * ``dic['rings block']`` (int,optional): Blocks of sinograms to be used. Even values between [1,20] [default: 1]   

    References:

        .. [1] Miqueles, E.X., Rinkel, J., O'Dowd, F. and Bermudez, J.S.V. (2014). Generalized Titarenko\'s algorithm for ring artefacts reduction. J. Synchrotron Rad, 21, 1333-1346. DOI: https://doi.org/10.1107/S1600577514016919
    
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