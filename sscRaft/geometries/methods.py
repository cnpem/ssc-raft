# Authors: Giovanni L. Baraldi, Gilberto Martinez
from ..rafttypes import *
from ..processing.io import *
from .parallel.fbp_methods import *
from .parallel.em_methods import *

def fbp(tomogram, dic, angles = None, **kwargs):
    """Computes the reconstruction of a parallel beam tomogram using the 
    Backprojection Slice Theorem (BST) method.
    

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        dic (dict): Dictionary with the experiment info.

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

    * One or MultiGPUs. 
    * Calls function ``bstGPU()``
    * Calls function ``fbpGPU()``


    Dictionary parameters:

        * ``dic['gpu']`` (ndarray): List of gpus  [required]
        * ``dic['angles[rad]']`` (list): List of angles in radians [required]
        * ``dic['method']`` (str,optional):  [Default: 'RT']

             #. Options = (\'RT\',\'BST\')

        * ``dic['filter']`` (str,optional): Filter type [Default: \'lorentz\']

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')
          
        * ``dic['beta/delta']`` (float,optional): Paganin by slices method ``beta/delta`` ratio [Default: 0.0 (no Paganin applied)]
        * ``dic['z2[m]']`` (float,optional): Sample-Detector distance in meters used on Paganin by slices method. [Default: 0.0 (no Paganin applied)]
        * ``dic['energy[eV]']`` (float,optional): beam energy in eV used on Paganin by slices method. [Default: 0.0 (no Paganin applied)]
        * ``dic['regularization']`` (float,optional): Regularization value for filter ( value >= 0 ) [Default: 1.0]
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [Default: 2]

    References:

        .. [1] Miqueles, X. E. and Koshev, N. and Helou, E. S. (2018). A Backprojection Slice Theorem for Tomographic Reconstruction. IEEE Transactions on Image Processing, 27(2), p. 894-906. DOI: https://doi.org/10.1109/TIP.2017.2766785.
    
    """
    required = ('gpu',)        
    optional = ( 'filter','offset','padding','regularization','beta/delta','blocksize','energy[eV]','z2[m]','method')
    default  = ('lorentz',       0,        2,             1.0,         0.0,          0,         1.0,    1.0,    'RT')
    
    dic = SetDictionary(dic,required,optional,default)

    dic['regularization'] = 1.0
    method                = dic['method']
    gpus                  = dic['gpu']

    if method == 'RT':
        try:
            angles = dic['angles[rad]']
        except:
            if angles is None:
                logger.error(f'Missing angles list!! Finishing run...') 
                raise ValueError(f'Missing angles list!!')

        output = fbpGPU( tomogram, angles, gpus, dic ) 

        return output
    
    if method == 'BST':

        angles = numpy.linspace(0.0, numpy.pi, tomogram.shape[-2], endpoint=False)

        output = bstGPU( tomogram, angles, gpus, dic ) 
    

        return output

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
