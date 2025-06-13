# Authors: Giovanni L. Baraldi, Gilberto Martinez
from ..rafttypes import *
from ..io.io_ import *

def lowpass(tomogram, dic = None, **kwargs):
    """Low pass filter of the Filtered BackProjection method.
    

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        dic (dict, optional): Dictionary with the experiment info [default: None

    Returns:
        (ndarray): Filtered tomogram. The axes are [slices, angles, lenght]

    Dictionary parameters:

        * ``dic['gpu']`` (ndarray): List of gpus  [required]
        * ``dic['filter']`` (str,optional): Filter type [Default: \'ramp\']

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')
        
        * ``dic['detectorPixel[m]']`` (float,optional): Detector pixel size in meters [Default: 1.0]
        * ``dic['beta/delta']`` (float,optional): Paganin by slices method ``beta/delta`` ratio [Default: 0.0 (no Paganin applied)]
        * ``dic['z2[m]']`` (float,optional): Source-Detector divided by Sample-Detector distance used on Paganin by slices method. [Default: 1.0]
        * ``dic['energy[eV]']`` (float,optional): beam energy in eV used on Paganin by slices method. [Default: 1.0 ]
        * ``dic['regularization']`` (float,optional): Regularization value for some filters ( value >= 0 ) [Default: 0.0]

            #. Related filters: \'gaussian\', \'lorentz\' and \'rectangle\'

        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [Default: 2]
        * ``dic['blocksize']`` (int,optional): Block of slices to be simultaneously computed [Default: 0 (automatic)]

    """
    required = ('gpu',)        
    optional = ('filter','padding','regularization','beta/delta','blocksize','energy[eV]','z2[m]','detectorPixel[m]', 'magnitude')
    default  = (  'ramp',        2,             0.0,         0.0,          0,         1.0,    1.0,               1.0,         1.0)
    
    dic      = SetDictionary(dic,required,optional,default)  

    ngpus    = len(dic['gpu'])
    gpus     = numpy.array(dic['gpu'])
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = tomogram.shape[-1]
    nangles  = tomogram.shape[-2]
    
    if len(tomogram.shape) == 2:
        nslices = 1
    else:
        nslices = tomogram.shape[0]
    
    filter_type    = FilterNumber(dic['filter'])
    beta_delta     = dic['beta/delta']
    regularization = dic['regularization']
    offset         = 0.0
    blocksize      = dic['blocksize']
    energy         = dic['energy[eV]']
    z2             = dic['z2[m]']
    pixelx, pixely = dic['detectorPixel[m]'],dic['detectorPixel[m]']
    wavelength     = CONST/energy 

    if beta_delta != 0.0:
        beta_delta = 1.0 / beta_delta
        paganin_slices_regularization = wavelength * z2 * numpy.pi * beta_delta / (pixelx * pixelx); 
    else:
        paganin_slices_regularization = 0.0

    padx = dic['padding']

    tomo_size    = dim3(x =   nrays, y = nangles, z = nslices)
    tomo_pad     = dim3(x = padx, y =    0, z = 0)
    tomo_dim     = DIM(size = tomo_size, pad = tomo_pad, blocksize = blocksize)

    geometry     = GEO(detector_pixel_x = pixelx, detector_pixel_y = pixely, 
                       obj_pixel_x = pixelx, obj_pixel_y = pixelx, 
                       energy = energy, wavelength = wavelength, 
                       z1x = 0, z1y = 0, z2x = z2, z2y = z2, 
                       magnitude_x = 1.0, magnitude_y = 1.0)
    
    ReconParam   = REC( method = 0, filter = filter_type, filter_reg = regularization,
                        beta_delta = paganin_slices_regularization, iterations = 0, 
                        rotation_axis_offset = offset,
                        total_variation = 0, interpolation = 0)

    tomogram     = CNICE(tomogram) 
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    libraft.getFilterLowPassMultiGPU(tomo_dim, geometry, ReconParam,
                                     gpus_ptr, ctypes.c_int(ngpus), tomogram_ptr)
    
    return tomogram
