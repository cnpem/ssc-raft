from ..rafttypes import *
from ..io.io_ import *

def phase_retrieval(frames, dic):
    """ Application of phase retrieval methods based on the Transport of Equation (TIE) approach [1]_, and/or 
    contrast filters methods.
    The data measurement needs to be corrected by flat (or empty) and dark previously.
    Flat is a measurement without a sample, to measure the background. 
    Dark is a measurement without a beam and sample, to measure detector pixel response.
    
    The data input can be computed as

    .. math::
        T = \\frac{D - D_d}{D_f - D_d}

    where :math:`T` is the corrected frames, :math:`D` is the measurements volume, 
    :math:`D_f` is the flat measurement and :math:`D_d` is the dark measurement.

    The logarithm can be applied or not on the output data, as in [1]_, depending on the ``dic['post_process']`` parameter.

    Args:
        frames (ndarray): 2D or 3D tomogram data. Axes are [angles,slices,rays].
        dic (dict): dictionary with function parameters.

    Returns:
        (ndarray): 2D or 3D filtered tomogram. Axes are [angles,slices,rays].

    Dictionary parameters:

        * ``dic['gpu']`` (list of ints): List of GPUs. Example [0,1,2] for 3 GPUs [default: [0]] 
        * ``dic['method']`` (str): Method - options: \'paganin\', \'contrast\' [default: \'paganin\']
        * ``dic['beta/delta']`` (float): Paganin ``beta/delta`` ratio [default: 0.0] 
        * ``dic['z2[m]']`` (float): Sample-Detector distance in meters [default: 1.0] 
        * ``dic['detectorPixel[m]']`` (float): Detector pixel size in meters [default: 1.0] 
        * ``dic['energy[eV]']`` (float): Beam line energy in KeV [default: 1.0] 
        * ``dic['magn']`` (float): Beam magnification [default: 1.0] 
        * ``dic['regularization']`` (float,optional): Regularization parameter for \'contrast\' method [default: 0.0] 
        * ``dic['post_process']`` (bool,optional): Apply post kernel function. Ex: apply ``-log()`` after Paganin kernel [default: False]
        * ``dic['padding']`` (float,optional): Data padding - percentage of data size (0.1,0.5,1.0, etc...) [default: 0.25]
        * ``dic['padd_mode']`` (str,optional): Data padding mode - options: \'none\', \'zero\', \'ones\', \'edge\' [default: \'edge\'] 
        * ``dic['blocksize']`` (int,optional): Size of projection blocks to be processed simultaneously [default: 0 (automatic computation)]
 
    References:

        .. [1] D. Paganin, S. C. Mayo, T. E. Gureyev, P. R. Miller, S. W. Wilkins (2002). Simultaneous phase and amplitude extraction from a single defocused image of a homogeneous object. Journal of Microscopy, 206:33-40. DOI: https://doi.org/10.1046/j.1365-2818.2002.01010.x

    """  
    required = None # ('required',)
    optional = ('gpu', 'method','beta/delta','padding','blocksize','z2[m]','energy[eV]','magn','detectorPixel[m]','regularization', 'post_process')
    default  = (  [0],'paganin',         0.0,     0.25,          0,    1.0,         1.0,   1.0,               1.0,             0.0,          False)
    
    dic      = SetDictionary(dic,required,optional,default)

    gpus     = dic.get('gpu', [0])
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = frames.shape[-1]
    nslices  = frames.shape[-2]

    if len(frames.shape) == 2:
        nangles = 1
    else:
        nangles = frames.shape[0]

    beta_delta = dic.get('beta/delta', 0.0)
    z2         = dic.get('z2[m]', 1.0)
    energy     = dic.get('energy[eV]', 1.0)
    magn       = dic.get('magn', 1.0)
    pixel_det  = dic.get('detectorPixel[m]', 1.0)
    wavelength = CONST/energy 
    pixel_obj  = pixel_det / magn
    reg        = dic.get('regularization', 0.0)
    padding    = dic.get('padding', 0.25)*100 # Multiply by 100 to get an integer value
    padd_mode  = PaddMode(dic.get('padd_mode', 'edge'))
    blocksize  = dic.get('blocksize', 0)
    post_proc  = dic.get('post_process', False)

    if post_proc is False:
        post_proc = 0
    else:
        post_proc = 1

    if blocksize > ( nangles // ngpus ):
        logger.error(f'Blocksize is bigger than the number of angles ({nangles}) divided by the number of GPUs selected ({ngpus})!')
        raise ValueError(f'Blocksize is bigger than the number of angles ({nangles}) divided by the number of GPUs selected ({ngpus})!')

    methodname      = dic['method']
    method          = ContrastFilterNumber(methodname)

    tomo_dim        = dimension((nrays, nslices, nangles), (padding, padding, 0), blocksize = blocksize, padd_mode = padd_mode)
    
    geometry        = define_geometry(detector_pixel = (pixel_det, pixel_det),
                                      obj_pixel      = (pixel_obj, pixel_obj),
                                      z1             = (0,0),
                                      z2             = (z2,z2),
                                      magnitude      = (magn,magn), 
                                      energy         = energy, 
                                      wavelength     = wavelength)
    
    Params          = contrast_param(method, beta_delta, regularization = reg, post_process = post_proc)
       
    frames          = CNICE(frames, numpy.float32)
    frames_ptr      = frames.ctypes.data_as(ctypes.c_void_p)

    libraft.getContrastEnhencementMultiGPU(tomo_dim, geometry, Params, 
                             gpus_ptr, ctypes.c_int(ngpus), frames_ptr)                     

    return frames