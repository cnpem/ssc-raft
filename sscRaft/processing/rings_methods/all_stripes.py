import numpy as np
from ...rafttypes import *
from ...io import SetDictionary

def rings_remove_all_stripes(tomo, ngpus = 1, settings = None, batch = 256):
    """Remove rings artifacts with all stripes method [1]_.
    
    Args:
        tomo (ndarray): Tomogram. The axes are [slices, angles, lenght].
        ngpus (int, optional):  Number of GPUs to be used [default: 1]
        settings (dict, optional): Dictionary with the parameters [default: None]
        batch (int, optional): Number of tomogram slice blocks for simultaneous computation [default: 256]

    Returns:
        (ndarray): Filtered tomogram 3D object. The axes are [slices, angles, lenght]

    * One or MultiGPUs. 

    Dictionary parameters:

        * ``settings['size_median']`` (int,optional): Window size of the median filter to remove small-to-medium stripes [Default: 21]
        * ``settings['large_stripes_size']`` (int,optional): Window size of the median filter to remove large stripes [Default: 61]
        * ``settings['background_ratio']`` (float,optional): Signal to noise ratio. Used to locate large stripes or bad pixels stripes. Greater is less sensitive to noise [Default: 10.0]
        * ``settings['drop_ratio']`` (float,optional): Ratio of pixels to be dropped, which is used to reduce the false detection of stripes [Default: 0.1]
        * ``settings['dim']`` (int,optional): Dimension of the window. Only supports 1 [default: 1]

    References:

        .. [1] Macul, L. and Miqueles, E.X. (2025). Accelerating Tomographic Artifact Removal Using a Multi-GPU System. Communications in Computer and Information Science, vol 2270. Springer, Cham. DOI: https://doi.org/10.1007/978-3-031-80084-9_1.

    """
    # Set default dictionary parameters:
    required = None
    optional = ('size_median','large_stripes_size','background_ratio','drop_ratio','dim')
    default  = (           21,                  61,             10.0,         0.1,     1)

    settings = SetDictionary(settings,required,optional,default)

    tomo_attributes = set_all_tripes_attributes(tomo.shape, settings)
    # if tomo.dtype != np.float32:
    #     tomo = tomo.astype('float32')
    # if not tomo.data.c_contiguous:
    #     tomo = np.ascontiguousarray(tomo)

    tomo     = CNICE(tomo)
    tomo_ptr = tomo.ctypes.data_as(ctypes.c_void_p)

    libraft.remove_all_stripes(tomo_ptr, tomo_attributes, ctypes.c_int(ngpus), ctypes.c_int(batch))

    return np.reshape(tomo, (tomo_attributes.n_slice, tomo_attributes.n_angle, tomo_attributes.n_length))

def rings_remove_all_stripes_multiaxis(tomo, ngpus = 1, settings = None, batch = 256):
    """Remove rings artifacts with all stripes multiaxis method.
    
    Args:
        tomo (ndarray): Tomogram. The axes are [slices, angles, lenght].
        ngpus (int, optional):  Number of GPUs to be used [default: 1]
        settings (dict, optional): Dictionary with the parameters [default: None]
        batch (int, optional): Number of tomogram slice blocks for simultaneous computation [default: 256]

    Returns:
        (ndarray): Filtered tomogram 3D object. The axes are [slices, angles, lenght]

    * One or MultiGPUs. 

    Dictionary parameters:

        * ``settings['size_median']`` (int,optional): Window size of the median filter to remove small-to-medium stripes [Default: 21]
        * ``settings['large_stripes_size']`` (int,optional): Window size of the median filter to remove large stripes [Default: 61]
        * ``settings['background_ratio']`` (float,optional): Signal to noise ratio. Used to locate large stripes or bad pixels stripes. Greater is less sensitive to noise [Default: 10.0]
        * ``settings['drop_ratio']`` (float,optional): Ratio of pixels to be dropped, which is used to reduce the false detection of stripes [Default: 0.1]
        * ``settings['dim']`` (int,optional): Dimension of the window. Only supports 1 [default: 1]

    """
    # Set default dictionary parameters:
    required = None
    optional = ('size_median','large_stripes_size','background_ratio','drop_ratio','dim')
    default  = (           21,                  61,             10.0,         0.1,     1)

    settings = SetDictionary(settings,required,optional,default)    
    tomo_attributes = set_all_tripes_attributes(tomo.shape, settings)
    # if tomo.dtype != np.float32:
    #     tomo = tomo.astype('float32')
    # if not tomo.data.c_contiguous:
    #     tomo = np.ascontiguousarray(tomo)

    tomo     = CNICE(tomo)
    tomo_ptr = tomo.ctypes.data_as(ctypes.c_void_p)

    libraft.remove_all_stripes_transposed(tomo_ptr, tomo_attributes, ctypes.c_int(ngpus), ctypes.c_int(batch))

    return np.reshape(tomo, (tomo_attributes.n_slice, tomo_attributes.n_angle, tomo_attributes.n_length))


