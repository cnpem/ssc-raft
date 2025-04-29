from ...rafttypes import *
import numpy

def fdk(tomogram: numpy.ndarray, dic: dict = {}, angles: numpy.ndarray = None, obj: numpy.ndarray = None) -> numpy.ndarray:
    """Computes the Reconstruction of a Conical Sinogram using the Filtered Backprojection method for conical rays(FDK).
    GPU function.

    Args:
        data (ndarray): Cone beam projection tomogram. The axes are [slices, angles, lenght].
        dic (dictionary): Dictionary with the parameters.

    Returns:
        (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [z, y, x].

    Dictionary parameters:

        * ``dic['gpu']`` (ndarray): List of gpus for processing [required]
        * ``dic['angles[rad]']`` (list): List of angles in radians [required]
        * ``dic['z1[m]']`` (float): Source-sample distance in meters [required]
        * ``dic['z1+z2[m]']`` (float): Source-detector distance in meters [required]
        * ``dic['detectorPixel[m]']`` (float): Detector pixel size in meters [required]
        * ``dic['filter']`` (str,optional): Filter type [Default: \'lorentz\']

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')
          
        * ``dic['beta/delta']`` (float,optional): Paganin by slices method ``beta/delta`` ratio [Default: 0.0 (no Paganin applied)]
        * ``dic['energy[eV]']`` (float,optional): beam energy in eV used on Paganin by slices method. [Default: 0.0 (no Paganin applied)]
        * ``dic['regularization']`` (float,optional): Regularization value for filter ( value >= 0 ) [Default: 1.0]
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [Default: 2]
        * ``dic['blocksize']`` (int,optional): Block of slices to be simultaneously computed [Default: 0 (automatic)]

    """
    try:
        blocksize = dic['blocksize']
    except:
        blocksize = 0
    # recon = data 
    try:
        regularization = dic['beta/delta']

        if regularization != 0.0:
            regularization = 1.0 / regularization
    except:
        regularization = 0.0

    Dd, Dsd = dic['z1[m]'], dic['z1+z2[m]']
    dh, dv  = dic['detectorPixel[m]'], dic['detectorPixel[m]']
    energy  = dic['energy[eV]']

    nrays   = tomogram.shape[-1]
    nangles = tomogram.shape[-2]

    if len(tomogram.shape) == 2:
            nslices = 1

    elif len(tomogram.shape) == 3:
            nslices = tomogram.shape[0]
    else:
        message_error = f'Data has wrong shape = {tomogram.shape}. It needs to be 2D or 3D.'
        logger.error(message_error)
        raise ValueError(message_error)
    
    try:
        padh  = dic['padding']
    except:
        padh  = 0

    if angles is None:
        try:
            angles = dic['angles[rad]']
        except:
            message_error = f'Missing angles list.'
            logger.error(message_error) 
            raise ValueError(message_error)
    
    start_recon_slice = 0
    end_recon_slice   = nslices
    is_slice          = 0
    logger.info(f'Reconstruct all slices: {start_recon_slice} to {end_recon_slice}.')  

    nx, ny, nz = int(nrays), int(nrays), int(nslices)
    logger.info(f'Set default reconstruction size ({nx},{ny},{nz}) = (x,y,z).')

    start_tomo_slice  = 0
    end_tomo_slice    = nslices

    nh, nv = int(nrays), int(nslices)
    h, v   = nh*dh/2, nv*dv/2
    nph    = int( nh * ( 1 + padh ) )

    print("nph:",nph,padh)

    nbeta  = len(angles)

    if nbeta != nangles: 
        message_error = f'Number of projection do not match: size of angles list ({nbeta}) is different from the number of projections ({nangles}).'
        logger.error(message_error)
        raise ValueError(message_error)

    beta_max = angles[nbeta - 1]
    dbeta    = angles[1] - angles[0]

    magn       = Dd/Dsd
    dx, dy, dz = dh*magn, dh*magn, dv*magn
    x, y, z    = dx*nx/2, dy*ny/2, dz*nslices/2

    fourier    = True
    filtername = dic['filter']
    filter     = FilterNumber(dic['filter'])

    logger.info(f'FDK filter: {filtername}({filter})')

    offset = 0 # dic['rotation axis offset'] -> Bug as of 22/Ago/2024 Paola Ferraz
    
    lab = Lab(  x = x, y = y, z = z, 
                dx = dx, dy = dy, dz = dz, 
                nx = nx, ny = ny, nz = nz, 
                h = h, v = v, 
                dh = dh, dv = dv, 
                nh = nh, nv = nv, 
                D = Dd, Dsd = Dsd, 
                beta_max = beta_max, 
                dbeta = dbeta, 
                nbeta = nbeta,
                fourier = fourier, 
                filter_type = filter, 
                reg = regularization,
                is_slice = is_slice,
                slice_recon_start = start_recon_slice, slice_recon_end = end_recon_slice,  
                slice_tomo_start = start_tomo_slice, slice_tomo_end = end_tomo_slice,
                nph = nph, padh = padh,
                energy = energy, rotation_axis_offset = offset,
                blocksize = blocksize)

    time = numpy.zeros(2)
    time = numpy.ascontiguousarray(time.astype(numpy.float64))
    time_p = time.ctypes.data_as(ctypes.c_void_p)

    gpus = numpy.array(dic['gpu'])
    ndev = len(gpus)
    gpus = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_p = gpus.ctypes.data_as(ctypes.c_void_p)

    angles = CNICE(angles, numpy.float32)
    angles_p = angles.ctypes.data_as(ctypes.c_void_p)

    logger.info(f'Tomogram data shape: {tomogram.shape} = (slices,angles,rays).')

    proj = CNICE(tomogram, numpy.float32)
    proj_p = proj.ctypes.data_as(ctypes.c_void_p)

    logger.info(f'Recon shape: ({lab.nx}, {lab.ny}, {lab.nz}) = (nx,ny,nz).')

    if obj is None:
        obj = numpy.zeros([lab.nz, lab.ny, lab.nx], dtype=numpy.float32)  
        obj = CNICE(obj)
    obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    libraft.gpu_fdk(lab, obj_ptr, proj_p, angles_p, gpus_p, 
                    ctypes.c_int(ndev), time_p)

    return obj
