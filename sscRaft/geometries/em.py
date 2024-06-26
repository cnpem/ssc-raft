from ..rafttypes import *

from .parallel.em import *
from ..processing.io import *

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
    * Calls function ``eEMRT_GPU_()``.
    * Calls function ``tEMRT_GPU_()``.
    * Calls function ``tEMFQ_GPU_()``.
    
    Dictionary parameters:
    
        * ``dic['gpu']`` (int list):  List of GPU devices used for computation [required]
        * ``dic['flat']`` (ndarray):  Flat 2D data. Tha axis are (slices,rays)  [required]
        * ``dic['angles[rad]']`` (floar list):  List of angles in radians [required]
        * ``dic['detectorPixel[m]']`` (float): Detector pixel size in meters [required for ``tEMFQ``]
        * ``dic['method']`` (str): Choose EM-method. Options: [required]
    
            #. ``eEMRT``: Emission EM using Ray Tracing as forward and inverse operators.

            #. ``tEMRT``: Transmission EM using Ray Tracing as forward and inverse operators.

            #. ``tEMFQ``: Transmission EM using the Fourier Slice Theorem (FST) for the forward operator and Backprojection Slice Theorem (BST) for the inverse operator.
        
        * ``dic['beamgeometry']`` (str): Beam geometry - \'parallel\', \'conebeam\' or \'fanbeam`\' [default: \'parallel\'] [required]
        * ``dic['iterations']`` (int, optional): Global number of iterations [default: 100]
        * ``dic['interpolation']`` (str, optional):  Type of interpolation. Options: \'nearest\' or \'bilinear\' [default: \'bilinear\']
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 2]  

    """
    # Set default dictionary parameters:
    required = ('gpu',)
    optional = ('iterations','detectorPixel[m]','padding','beamgeometry','interpolation','blocksize')
    default  = (10,0.0,2,'parallel','bilinear',0)

    dic          = SetDictionary(dic,required,optional,default)

    blocksize     = dic['blocksize']

    gpus          = dic['gpu']
    method        = dic['method']

    iterations    = dic['iterations']
    TV_iterations = 0 #dic['TV iterations']

    det_pixel     = (dic['detectorPixel[m]'],dic['detectorPixel[m]']) # det_pixel = (det_pixelx, det_pixely)
    pad           = (dic['padding'],0,0) # pad = (padx, pady, padz)

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
        
    if method != 'eEMRT':
        try:
            flat = dic['flat']
        except:
            if flat is None:
                logger.warning(f'No flat provided.') 
                flat = numpy.ones((data.shape[0],data.shape[-1])) 

    # Initial guess for EM Frequency
    try:
        guess = dic['guess']
    except:
        pass 

    if method == 'eEMRT':

        output = eEMRT_GPU_(data, angles, iterations, gpus, blocksize) 
    
    elif method == 'tEMRT':

        output = tEMRT_GPU_(data, flat, angles, iterations, gpus, blocksize)

    elif method == 'tEMFQ':

        output = tEMFQ_GPU_(data, flat, angles, 
                            pad, interpolation, det_pixel, 
                            tv_reg, iterations, gpus, blocksize, guess)  
    else:
        logger.error(f'Invalid EM method:{method}')
        raise ValueError(f'Invalid EM method:{method}')

    return output


