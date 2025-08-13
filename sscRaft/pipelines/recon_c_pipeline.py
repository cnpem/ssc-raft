from ..rafttypes import *

def FastReconPipeline(tomogram, flat, dark, angles = None, gpus = [0], dic = None, obj = None, **kwargs):
    """Wrapper fo MultiGPU/CUDA function that computes the reconstruction pipeline in ``C`` and ``CUDA``.

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        flat (ndarray): The flat image for correction.
        dark (ndarray): The dark image for correction.
        angles (float list): List of angles in radians
        gpus (int list): List of gpus
        dic (dict): Dictionary with parameters info

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

    """
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = CNICE(gpus)
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = tomogram.shape[-1]
    nangles  = tomogram.shape[-2]
    nslices  = tomogram.shape[0] if tomogram.ndim > 2 else 1
    objsize  = nrays
    nflats   = flat.shape[0] if flat.ndim > 2 else 1
    print('Python nflats:',nflats)
    print('Python nflats.ndim:',flat.ndim)
    if dark.ndim > 2:
        dark = dark[0]

    if obj is None:
        obj = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
        obj = CNICE(obj)
    else:
        objsize = obj.shape[-1]

    filter_type    = FilterNumber(dic.get('filter', 'ramp'))
    beta_delta     = dic.get('beta/delta', 0.0)
    regularization = dic.get('regularization', 1.0)
    offset         = dic.get('rotation axis offset', 0)
    blocksize      = dic.get('blocksize', 0)
    energy         = dic.get('energy[eV]', 1.0)
    z2             = dic.get('z2[m]', 1.0)
    pixel          = dic.get('detectorPixel[m]', 1.0)
    wavelength     = CONST/energy 
    padding        = dic.get('padding', 0.25)*100 # Multiply by 100 to get an integer value
    padd_mode      = PaddMode(dic.get('padd_mode', 'edge'))

    do_norm  = dic.get('do_norm', 1)
    do_log   = dic.get('do_log', 1)
    do_rings = dic.get('do_rings', 0)
    do_recon = dic.get('do_recon', 0)

    # Paganin by Slices here
    if beta_delta != 0.0:
        beta_delta = 1.0 / beta_delta
        paganin_slices_regularization = wavelength * z2 * numpy.pi * beta_delta / (pixel * pixel); 
    else:
        paganin_slices_regularization = 0.0

    tomo_dim  = dimension((  nrays, nangles, nslices), (padding,       0, 0), blocksize = blocksize, padd_mode = padd_mode)
    obj_dim   = dimension((objsize, objsize, nslices), (padding, padding, 0), blocksize = blocksize, padd_mode = padd_mode)
    
    geometry  = define_geometry(detector_pixel = (pixel, pixel),
                                obj_pixel      = (pixel, pixel),
                                z1             = (0,0),
                                z2             = (z2,z2),
                                magnitude      = (1.0,1.0), 
                                energy         = energy, 
                                wavelength     = wavelength)
    
    Contrast = contrast_param(method = 0, beta_delta = beta_delta, regularization = 1.0, post_process = 0)

    Rings    = rings_param(method = 0, rings_block = 2, rings_lambda = -1)

    Recon    = REC(method               = 1, # FBP_RT
                   filter               = filter_type, 
                   filter_reg           = regularization,  
                   paganin_slices       = paganin_slices_regularization, 
                   iterations           = 0, 
                   rotation_axis_offset = offset,
                   total_variation      = 0, 
                   interpolation        = 0)

    Flags    = FLAG(do_flat_dark_correction = do_norm,
                    do_flat_dark_log        = do_log,
                    do_paganin_filter       = 0,
                    do_rings                = do_rings,
                    do_rotation_axis_offset = 0,
                    do_rotation_correction  = 0,
                    do_alignment            = 0,
                    do_excentric            = 0,
                    do_reconstruction       = do_recon)

    Configs  = configs_param(tomo_dim       = tomo_dim, 
                             obj_dim        = obj_dim, 
                             geometry       = geometry, 
                             rings_param    = Rings, 
                             contrast_param = Contrast, 
                             recon_param    = Recon, 
                             nflats         = nflats, 
                             flags          = Flags)

    angles       = numpy.array(angles)
    angles       = CNICE(angles) 
    angles_ptr   = angles.ctypes.data_as(ctypes.c_void_p) 

    flat       = CNICE(flat)
    flat_ptr   = flat.ctypes.data_as(ctypes.c_void_p)

    dark       = CNICE(dark)
    dark_ptr   = dark.ctypes.data_as(ctypes.c_void_p)
    
    tomogram     = CNICE(tomogram)
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)
    obj_ptr      = obj.ctypes.data_as(ctypes.c_void_p)

    libraft.ReconstructionPipelineMultiGPU(Configs, gpus_ptr, ngpus,
                                           obj_ptr, tomogram_ptr, 
                                           flat_ptr, dark_ptr, angles_ptr)

    return obj, tomogram
