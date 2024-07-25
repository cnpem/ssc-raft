import numpy
from ..rafttypes import *

RECON_METHODS = {
    'FBP': 0,
    'BST': 1
}

def reconPipeline(tomogram, flat, dark, angles, gpus, dic):
    """Wrapper fo MultiGPU/CUDA function that computes the reconstruction of a parallel beam
    tomogram using the Filtered Backprojection (FBP) method.

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        flat (ndarray): The flat image for correction.
        dark (ndarray): The dark image for correction.
        angles (float list): List of angles in radians
        gpus (int list): List of gpus
        dic (dict): Dictionary with parameters info

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

    Dictionary parameters:

        * ``dic['filter']`` (str): Filter type [required]

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')

        * ``dic['beta/delta']`` (float,optional): Paganin by slices method ``beta/delta`` ratio [Default: 0.0 (no Paganin applied)]
        * ``dic['z2[m]']`` (float,optional): Sample-Detector distance in meters used on Paganin by slices method. [Default: 0.0 (no Paganin applied)]
        * ``dic['energy[eV]']`` (float,optional): beam energy in eV used on Paganin by slices method. [Default: 0.0 (no Paganin applied)]
        * ``dic['regularization']`` (float,optional): Regularization value for filter ( value >= 0 ) [Default: 1.0]
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [Default: 2]

    """
    ngpus = len(gpus)
    gpus = numpy.array(gpus, dtype=np.int32)
    gpus = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    *_, nangles, nrays = tomogram.shape

    nslices = tomogram.shape[0] if tomogram.ndim >= 2 else 1

    objsize = nrays

    filter_type = FilterNumber(dic['filter'])
    beta_delta = dic['beta/delta']
    regularization = dic['regularization']
    offset = dic['offset']
    blocksize = dic['blocksize']

    if beta_delta != 0.0:
        beta_delta = 1.0 / beta_delta
        energy = dic['energy[eV]']
        z2 = dic['z2[m]']
    else:
        beta_delta = 0.0
        energy = 1.0
        z2 = 0.0

    padx, pady, padz = dic['padding'], dic['padding'], 0  # (padx, pady, padz)

    tomogram = CNICE(tomogram)
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    obj = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
    obj_ptr = obj.ctypes.data_as(ctypes.c_void_p)

    flat = numpy.array(flat)
    flat = CNICE(flat)
    flat_ptr = flat.ctypes.data_as(ctypes.c_void_p)

    dark = numpy.array(dark)
    dark = CNICE(dark)
    dark_ptr = dark.ctypes.data_as(ctypes.c_void_p)

    angles = numpy.array(angles)
    angles = CNICE(angles)
    angles_ptr = angles.ctypes.data_as(ctypes.c_void_p)

    nflats = 1 if flat.ndim <= 2 else flat.shape[0]

    phase_type = 0
    rings_block = dic.get('rings_block', 0.0)
    rings_lambda = dic.get('rings_lambda', 0.0)

    if dic['method'] not in RECON_METHODS.keys():
        raise ArgumentError("Invalid reconstruction method: ", dic['method'])

    reconstruction_method = RECON_METHODS[dic['method']]

    obj_start_slice = 0
    obj_end_slice = nslices

    tomo_start_slice = 0
    tomo_end_slice = nslices

    em_iter = 100

    param_int = [
        objsize, objsize, nslices,
        nrays, nangles, nslices,
        padx, pady, padz,
        nflats,
        phase_type,
        rings_block,
        offset,
        reconstruction_method,
        filter_type,
        obj_start_slice, obj_end_slice,
        tomo_start_slice, tomo_end_slice,
        em_iter
    ]
    param_int = numpy.array(param_int)
    param_int = CNICE(param_int, numpy.int32)
    param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

    z1, z2 = dic['z1[m]'], dic['z2[m]']

    if dic['beamgeometry'] == 'parallel':
        magnitude_x = magnitude_y = 1.0
    elif dic['beamgeometry'] == 'fanbeam':
        magnitude_x = (z1 + z2) / z1
        magnitude_y = 1.0
    elif dic['beamgeometry'] == 'conebeam':
        magnitude_x = magnitude_y = (z1 + z2) / z1
    else:
        raise ValueError('Invalid beam geometry')

    param_float = [
        dic['detector_pixel[m]'], dic['detector_pixel[m]'],
        dic['energy[eV]'],
        z1, z1,
        z2, z2,
        magnitude_x, magnitude_y,
        beta_delta, rings_lambda,
        regularization
    ]
    param_float = numpy.array(param_float)
    param_float = CNICE(param_float, numpy.float32)
    param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)

    do_flat_dark_correction = True
    do_log_correction = False
    do_rotation = do_rotation_correction = True

    flags = numpy.array([
        do_flat_dark_correction,
        do_log_correction,
        False, False,
        do_rotation, do_rotation_correction,
        False
    ])
    flags = CNICE(flags, dtype=np.int32)
    flags_ptr = flags.ctypes.data_as(ctypes.c_void_p)

    libraft.ReconstructionPipeline(obj_ptr, tomogram_ptr,
                                   flat_ptr, dark_ptr, angles_ptr,
                                   param_float_ptr, param_int_ptr, flags_ptr,
                                   gpus_ptr, ctypes.c_int(ngpus))

    return obj
