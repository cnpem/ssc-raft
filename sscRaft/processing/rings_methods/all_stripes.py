import numpy as np
from ...rafttypes import *

def set_tomo_attributes(size, settings):
    (n_slice, n_angle, n_length)    = size
    size_median                     = settings['size_median'] + 1 if settings['size_median'] % 2 == 0 else settings['size_median']
    dim                             = settings['dim']
    large_stripes_size              = settings['large_stripes_size'] + 1 if settings['large_stripes_size'] % 2 == 0 else settings['large_stripes_size']
    background_ratio                = settings['background_ratio']
    drop_ratio                      = settings['drop_ratio']

    return TomoAttributes( n_length, n_angle, n_slice,
                            size_median, dim, large_stripes_size,
                            background_ratio, drop_ratio)

def rings_remove_all_stripes(tomo, ngpus, settings, batch = 256):

    tomo_attributes = set_tomo_attributes(tomo.shape, settings)
    # if tomo.dtype != np.float32:
    #     tomo = tomo.astype('float32')
    # if not tomo.data.c_contiguous:
    #     tomo = np.ascontiguousarray(tomo)

    tomo     = CNICE(tomo)
    tomo_ptr = tomo.ctypes.data_as(ctypes.c_void_p)

    libraft.remove_all_stripes(tomo_ptr, tomo_attributes, ctypes.c_int(ngpus), ctypes.c_int(batch))

    return np.reshape(tomo, (tomo_attributes.n_slice, tomo_attributes.n_angle, tomo_attributes.n_length))

def rings_remove_all_stripes_multiaxis(tomo, ngpus, settings, batch = 256):
    
    tomo_attributes = set_tomo_attributes(tomo.shape, settings)
    # if tomo.dtype != np.float32:
    #     tomo = tomo.astype('float32')
    # if not tomo.data.c_contiguous:
    #     tomo = np.ascontiguousarray(tomo)

    tomo     = CNICE(tomo)
    tomo_ptr = tomo.ctypes.data_as(ctypes.c_void_p)

    libraft.remove_all_stripes_transposed(tomo_ptr, tomo_attributes, ctypes.c_int(ngpus), ctypes.c_int(batch))

    return np.reshape(tomo, (tomo_attributes.n_slice, tomo_attributes.n_angle, tomo_attributes.n_length))


