from ..rafttypes import *
from ..processing.alignments.rotationaxis import Centersino
from ..processing.alignments.excentric_tomo import getOffsetExcentricTomo

def compute_rotation_axis_offset(rotation_axis_offset,align_method,tomogram,flat,dark,angles):
    
    if rotation_axis_offset is None:
        dangle = angles[1] - angles[0]
        iangle = ( np.pi / dangle ).astype(int)
        frame0 = tomogram[:,0,:]
        frame1 = tomogram[:,iangle-1,:]

        ft = 0*frame0 + 1
        dk = 0*frame0

        if flat.ndim > 2:
            ft = flat[:,0,:]

        if dark.ndim > 2:
            dk = dark[:,0,:]

        offset = Centersino(frame0, frame1, ft, dk)

        logger.info(f'Computed rotation axis offset: {offset}')
    else:
        offset = rotation_axis_offset

    if align_method == 'rot axis recon':
        offset_recon = offset
    else:
        offset_recon = 0

    return offset, offset_recon

def compute_excentric_offset(excentric_offset, tomogram, blocksize_excentric = 16):
    
    if excentric_offset is None:
        offset  = 0
        nslices = tomogram.shape[0]
        nhalf   = nslices // 2
        nblock  = blocksize_excentric // 2

        if nslices <= ( blocksize_excentric + 2 ):
            offset = getOffsetExcentricTomo(tomogram, gpus = [0])
        else:
            offset = getOffsetExcentricTomo(tomogram[nhalf - nblock:nhalf + nblock], gpus = [0])

        logger.info(f'Computed excentric offset: {offset}')
    else:
        offset = excentric_offset

    return offset


def FastReconPipeline(tomogram, flat, dark, angles = None, obj = None, gpus = [0], dic = None, **kwargs):
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
    gpus     = CNICE(gpus,numpy.intc)
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    angles   = sanitize_angles(numpy.array(angles))
    print('angles:',angles)

    nrays    = tomogram.shape[-1]
    nangles  = tomogram.shape[-2] # Put here a function to assess the dimension against angles list
    nslices  = tomogram.shape[0] if tomogram.ndim > 2 else 1
    objsizex = nrays
    objsizey = nrays
    nflats   = flat.shape[1] if flat.ndim > 2 else 1

    if dark.ndim > 2:
        dark = dark[1]

    recon_method   = dic.get('reconstruction method', 'fbp')
    rings_method   = dic.get('rings method', 'none')
    align_method   = dic.get('align method', 'rot axis recon')
    bckgC_method   = dic.get('background correction method', 'classic')

    filter_type    = dic.get('filter', 'ramp')
    beta_delta     = dic.get('beta/delta', 0.0)
    regularization = dic.get('regularization', 1.0)
    rotation_axis_offset = dic.get('rotation axis offset', None)
    excentric_offset = dic.get('excentric offset', None)
    blocksize      = dic.get('blocksize', 0)
    energy         = dic.get('energy[eV]', 1.0)
    z2             = dic.get('z2[m]', 1.0)
    pixel          = dic.get('detectorPixel[m]', 1.0)
    wavelength     = CONST/energy 
    iterations     = dic.get('iterations', 0)

    padding        = dic.get('padding', 1.0)*100 # Multiply by 100 to get an integer value
    padMode        = dic.get('padd_mode', 'zero')
    zoom_padding   = dic.get('zoom padding', 0.0)*100 # Multiply by 100 to get an integer value
    zoom_padd_mode = dic.get('zoom padd_mode', 'edge')

    median_s = dic.get('size_median',21)
    large_s  = dic.get('large_stripes_size',121)
    b_ratio  = dic.get('background_ratio',10.0)
    d_ratio  = dic.get('drop_ratio',0.1)
    d        = dic.get('dim',1)

    if align_method == 'excentric':
        objsizex, objsizey = get_excentric_obj_dimensions(objsizex, objsizey)
        excentric_offset   = compute_excentric_offset(excentric_offset, tomogram, blocksize_excentric = 16)

    rotation_axis_offset, rotation_axis_offset_recon = compute_rotation_axis_offset(rotation_axis_offset, align_method, tomogram, flat, dark, angles)

    flat_dim = dimension((   nrays,  nslices,  nflats))
    dark_dim = dimension((   nrays,  nslices,       1))
    tomo_dim = dimension((   nrays,  nangles, nslices), (zoom_padding,            0, 0), blocksize = blocksize, padd_mode = zoom_padd_mode)
    obj_dim  = dimension((objsizex, objsizey, nslices), (zoom_padding, zoom_padding, 0), blocksize = blocksize, padd_mode = zoom_padd_mode)
    
    geometry  = define_geometry(detector_pixel = (pixel, pixel, pixel),
                                obj_pixel      = (pixel, pixel, pixel),
                                z1             = 0.0,
                                z2             = z2,
                                magnitude      = 1.0, 
                                energy         = energy, 
                                wavelength     = wavelength)
    
    Contrast = contrast_param(method = 0, beta_delta = beta_delta, regularization = 1.0, post_process = 0)

    Rings    = rings_param(method       = rings_method, 
                           tomosize     = (nslices, nangles, nrays), 
                           median_s     = median_s, 
                           large_s      = large_s, 
                           b_ratio      = b_ratio, 
                           d_ratio      = d_ratio, 
                           d            = d, 
                           rings_block  = 2, 
                           rings_lambda = -1)

    Align    = align_param(method = align_method, excentric_offset = excentric_offset, rotation_axis_offset = rotation_axis_offset)

    Recon    = recon_param(geometry             = geometry, 
                           method               = recon_method, 
                           filter               = filter_type, 
                           filter_pad           = padding, 
                           filter_padMode       = padMode, 
                           beta_delta           = beta_delta, 
                           rotation_axis_offset = rotation_axis_offset_recon,
                           iterations           = iterations, 
                           filter_reg           = regularization,
                           total_variation      = 0.0, 
                           interpolation        = 'none')

    Configs  = configs_param(tomo_dim         = tomo_dim, 
                             obj_dim          = obj_dim,
                             flat_dim         = flat_dim,
                             dark_dim         = dark_dim, 
                             geometry         = geometry, 
                             rings_param      = Rings, 
                             contrast_param   = Contrast,
                             align_param      = Align, 
                             recon_param      = Recon,
                             back_corr_method = bckgC_method)
    
    if obj is None:
        obj = numpy.zeros([nslices, objsizex, objsizey], dtype=numpy.float32)
        obj = CNICE(obj)
    else:
        Objsizex = obj.shape[-1]
        Objsizey = obj.shape[-2]

        if ( Objsizex != objsizex ) or ( Objsizey != objsizey ):
            logger.error(f'Object size is not compatible with tomogram size')

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
    
    if align_method == 'excentric':
        tomogram = tomogram.reshape((nslices, nangles // 2, nrays * 2))

    return obj, tomogram

# def FastReconProcPipeline(tomogram, flat, dark, angles = None, gpus = [0], dic = None, obj = None, **kwargs):
#     """Wrapper fo MultiGPU/CUDA function that computes the reconstruction pipeline in ``C`` and ``CUDA``.

#     Args:
#         tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
#         flat (ndarray): The flat image for correction.
#         dark (ndarray): The dark image for correction.
#         angles (float list): List of angles in radians
#         gpus (int list): List of gpus
#         dic (dict): Dictionary with parameters info

#     Returns:
#         (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

#     """
#     libraft.ReconstructionPipelineProcessMultiGPU(  Configs, gpus_ptr, ngpus,
#                                                     obj_ptr, tomogram_ptr, 
#                                                     flat_ptr, dark_ptr, angles_ptr)

#     return obj, tomogram