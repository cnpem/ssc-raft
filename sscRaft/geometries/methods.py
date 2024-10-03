# Authors: Giovanni L. Baraldi, Gilberto Martinez
from ..rafttypes import *
from ..processing.io import *
from .parallel.fbp_methods import *
from .parallel.em_methods import *

def fbp(tomogram, angles = None, obj = None, dic = None, **kwargs):
    """Computes the reconstruction of a parallel beam tomogram using the 
    Backprojection Slice Theorem (BST) method [1]_.
    

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        angles (float list, optional):  List of angles in radians [default: None]
        obj (ndarray, optional): Reconstructed 3D object array [default: None]
        dic (dict, optional): Dictionary with the experiment info [default: None

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x]

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
        
        * ``dic['detectorPixel[m]']`` (float,optional): Detector pixel size in meters [Default: 1.0]
        * ``dic['beta/delta']`` (float,optional): Paganin by slices method ``beta/delta`` ratio [Default: 0.0 (no Paganin applied)]
        * ``dic['z2[m]']`` (float,optional): Sample-Detector distance in meters used on Paganin by slices method. [Default: 1.0]
        * ``dic['energy[eV]']`` (float,optional): beam energy in eV used on Paganin by slices method. [Default: 1.0 ]
        * ``dic['regularization']`` (float,optional): Regularization value for filter ( value >= 0 ) [Default: 1.0]
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [Default: 2]
        * ``dic['blocksize']`` (int,optional): Block of slices to be simulteneously computed [Default: 0 (automatically)]
        * ``dic['rotation axis offset']`` (int,optional): Rotation axis deviation value [Default: 0]


    References:

        .. [1] Miqueles, X. E. and Koshev, N. and Helou, E. S. (2018). A Backprojection Slice Theorem for Tomographic Reconstruction. IEEE Transactions on Image Processing, 27(2), p. 894-906. DOI: https://doi.org/10.1109/TIP.2017.2766785.
    
    """
    required = ('gpu',)        
    optional = ( 'filter','rotation axis offset','padding','regularization','beta/delta','blocksize','energy[eV]','z2[m]','method', 'detectorPixel[m]')
    default  = ('lorentz',                     0,        2,             1.0,         0.0,          0,         1.0,    1.0,    'RT',                1.0)
    
    dic = SetDictionary(dic,required,optional,default)

    method                = dic['method']
    gpus                  = dic['gpu']

    if method == 'RT':
        try:
            angles = dic['angles[rad]']
        except:
            if angles is None:
                logger.error(f'Missing angles list!! Finishing run...') 
                raise ValueError(f'Missing angles list!!')

        output = fbpGPU( tomogram, angles, gpus, dic, obj=obj)

        return output
    
    if method == 'BST':

        angles = numpy.linspace(0.0, numpy.pi, tomogram.shape[-2], endpoint=False)

        output = bstGPU( tomogram, angles, gpus, dic, obj=obj)

        return output

def em(data, flat = None, angles = None, obj = None, dic = None, **kwargs):
    """ Expectation maximization (EM) for 3D tomographic reconstructions for parallel, 
    conebeam and fanbeam geometries.

    Implemented methods for parallel geometry:

        * ``eEMRT``: Emission EM using Ray Tracing as forward and inverse operators.
        * ``tEMRT``: Transmission EM using Ray Tracing as forward and inverse operators.
        * ``tEMFQ``: Transmission EM using the Fourier Slice Theorem (FST) for the forward operator and Backprojection Slice Theorem (BST) for the inverse operator.

    Args:
        data (ndarray): Tomographic 3D data. The axis are (slices,angles,rays) 
        flat (ndarray, optional):  Flat 2D data. Tha axis are (slices,rays) [default: None]
        angles (float list, optional):  List of angles in radians [default: None]
        obj (ndarray, optional): Reconstructed 3D object array [default: None]
        dic (dict, optional): input dictionary [default: None]
        
    Returns:
        (ndarray): stacking 3D reconstructed volume, reconstructed sinograms (z,y,x)

    * One or MultiGPUs. 
    * Calls function ``eEMRT_GPU_()``.
    * Calls function ``tEMRT_GPU_()``.
    * Calls function ``tEMFQ_GPU_()``.
    
    Dictionary parameters:
    
        * ``dic['gpu']`` (int list):  List of GPU devices used for computation [required]
        * ``dic['detectorPixel[m]']`` (float): Detector pixel size in meters [required for ``tEMFQ``]
        * ``dic['method']`` (str): Choose EM-method. Options: [required]
    
            #. ``eEMRT``: Emission EM using Ray Tracing as forward and inverse operators.

            #. ``tEMRT``: Transmission EM using Ray Tracing as forward and inverse operators.

            #. ``tEMFQ``: Transmission EM using the Fourier Slice Theorem (FST) for the forward operator and Backprojection Slice Theorem (BST) for the inverse operator.
        
        * ``dic['beamgeometry']`` (str): Beam geometry - \'parallel\', \'conebeam\' or \'fanbeam`\' [default: \'parallel\'] [required]
        * ``dic['flat']`` (ndarray, optional):  Flat 2D data. Tha axis are (slices,rays) [default: None]
        * ``dic['angles[rad]']`` (float list, optional):  List of angles in radians [default: None]
        * ``dic['iterations']`` (int, optional): Global number of iterations [default: 100]
        * ``dic['interpolation']`` (str, optional):  Type of interpolation. Options: \'nearest\' or \'bilinear\' [default: \'bilinear\']
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 2]  
        * ``dic['blocksize']`` (int,optional): Block of slices to be simulteneously computed [default: 0 (automatically)]

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

    if angles is None:
        try:
            angles = dic['angles[rad]']
        except:
            logger.error(f'Missing angles list!! Finishing run...') 
            raise ValueError(f'Missing angles list!!')
        
    if method != 'eEMRT':
        if flat is None:
            try:
                flat = dic['flat']
            except:
                logger.warning(f'No flat provided.') 
                flat = numpy.ones((data.shape[0],data.shape[-1])) 

    if method == 'eEMRT':

        output = eEMRT_GPU_(data, angles, iterations, gpus, blocksize, obj = obj) 
    
    elif method == 'tEMRT':

        output = tEMRT_GPU_(data, flat, angles, iterations, gpus, blocksize, obj = obj)

    elif method == 'tEMFQ':

        output = tEMFQ_GPU_(data, flat, angles, 
                            pad, interpolation, det_pixel, 
                            tv_reg, iterations, gpus, blocksize, obj = obj)  
    else:
        logger.error(f'Invalid EM method:{method}')
        raise ValueError(f'Invalid EM method:{method}')

    return output
