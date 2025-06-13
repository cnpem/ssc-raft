from ..rafttypes import *
from ..io.io_ import *

def phase_retrieval(frames, dic):
    """ Application of phase retrieval methods based on the Transport of Equation (TIE) approach [1]_. 
    The data measurement needs to be corrected by flat (or empty) and dark previously.
    Flat is a measurement without a sample, to measure the background. 
    Dark is a measurement without a beam and sample, to measure detector pixel response.
    
    The data input can be computed as

    .. math::
        T = \\frac{D - D_d}{D_f - D_d}

    where :math:`T` is the corrected frames, :math:`D` is the measurements volume, 
    :math:`D_f` is the flat measurement and :math:`D_d` is the dark measurement.

    The logarithm is not applied on the output data, as in [1]_.

    Args:
        frames (ndarray): 2D or 3D tomogram data. Axes are [angles,slices,rays].
        dic (dict): dictionary with function parameters.

    Returns:
        (ndarray): 2D or 3D filtered tomogram. Axes are [angles,slices,rays].

    Dictionary parameters:

        * ``dic['gpu']`` (list of ints): List of GPUs. Example [0,1,2] for 3 GPUs [required] 
        * ``dic['beta/delta']`` (float): Paganin by slices method ``beta/delta`` ratio [default: 0.0] 
        * ``dic['method']`` (str): Method - options: \'paganin\' [default: \'paganin\']
        * ``dic['z2[m]']`` (float): Sample-Detector distance in meters [default: 1.0] 
        * ``dic['detectorPixel[m]']`` (float): Detector pixel size in meters [default: 1.0] 
        * ``dic['energy[eV]']`` (float): Beam line energy in KeV [default: 1.0] 
        * ``dic['magn']`` (float): Magnification of beam [default: 1.0] 
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 1] 
        * ``dic['blocksize']`` (int,optional): Size of projection blocks to be processed simultaneously [default: 0 (automatic computation)] 

    References:

        .. [1] D. Paganin, S. C. Mayo, T. E. Gureyev, P. R. Miller, S. W. Wilkins (2002). Simultaneous phase and amplitude extraction from a single defocused image of a homogeneous object. Journal of Microscopy, 206:33-40. DOI: https://doi.org/10.1046/j.1365-2818.2002.01010.x
    
    """  

    required = (    'gpu',)
    optional = ( 'method', 'beta/delta','padding','blocksize','z2[m]','energy[eV]','magn', 'detectorPixel[m]')
    default  = ('paganin',          0.0,        1,          0,      1,           1,     1,                1.0)
    
    dic = SetDictionary(dic,required,optional,default)

    gpus     = dic['gpu']     
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

    beta_delta = dic['beta/delta']
    z2         = dic['z2[m]']
    energy     = dic['energy[eV]']
    magn       = dic['magn']
    pixel_det  = dic['detectorPixel[m]']
    wavelength = CONST/energy 
    pixel_obj  = pixel_det / magn

    padx, pady = dic['padding'],dic['padding']

    blocksize = dic['blocksize']

    if blocksize > ( nangles // ngpus ):
        logger.error(f'Blocksize is bigger than the number of angles ({nangles}) divided by the number of GPUs selected ({ngpus})!')
        raise ValueError(f'Blocksize is bigger than the number of angles ({nangles}) divided by the number of GPUs selected ({ngpus})!')

    methodname      = dic['method']
    method          = PhaseMethodNumber(methodname)

    tomo_size    = dim3(x = nrays, y = nangles, z = nslices)
    tomo_pad     = dim3(x =  padx, y =    pady, z = 0)
    tomo_dim     = DIM(size = tomo_size, pad = tomo_pad, blocksize = blocksize)

    geometry     = GEO(detector_pixel_x = pixel_det, detector_pixel_y = pixel_det, 
                       obj_pixel_x = pixel_obj, obj_pixel_y = pixel_obj, 
                       energy = energy, wavelength = wavelength, 
                       z1x = 0, z1y = 0, z2x = z2, z2y = z2, 
                       magnitude_x = magn, magnitude_y = magn)
    
    ContrastEnhencementParam = CEF(method = method, beta_delta = beta_delta, paganin_lambda = 1.0)
   
    frames          = CNICE(frames, numpy.float32)
    frames_ptr      = frames.ctypes.data_as(ctypes.c_void_p)

    libraft.getContrastEnhencementMultiGPU(tomo_dim, geometry, ContrastEnhencementParam, 
                             gpus_ptr, ctypes.c_int(ngpus), frames_ptr)                     

    return frames
