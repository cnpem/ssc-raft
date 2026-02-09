import numpy as np
from ...rafttypes import *

def rings_remove_all_stripes(tomo, ngpus, settings, batch = 256):

    tomo_attributes = set_all_tripes_attributes(tomo.shape, settings)
    # if tomo.dtype != np.float32:
    #     tomo = tomo.astype('float32')
    # if not tomo.data.c_contiguous:
    #     tomo = np.ascontiguousarray(tomo)

    tomo     = CNICE(tomo)
    tomo_ptr = tomo.ctypes.data_as(ctypes.c_void_p)

    libraft.remove_all_stripes(tomo_ptr, tomo_attributes, ctypes.c_int(ngpus), ctypes.c_int(batch))

    return np.reshape(tomo, (tomo_attributes.n_slice, tomo_attributes.n_angle, tomo_attributes.n_length))

def rings_remove_all_stripes_multiaxis(tomo, ngpus, settings, batch = 256):
    
    tomo_attributes = set_all_tripes_attributes(tomo.shape, settings)
    # if tomo.dtype != np.float32:
    #     tomo = tomo.astype('float32')
    # if not tomo.data.c_contiguous:
    #     tomo = np.ascontiguousarray(tomo)

    tomo     = CNICE(tomo)
    tomo_ptr = tomo.ctypes.data_as(ctypes.c_void_p)

    libraft.remove_all_stripes_transposed(tomo_ptr, tomo_attributes, ctypes.c_int(ngpus), ctypes.c_int(batch))

    return np.reshape(tomo, (tomo_attributes.n_slice, tomo_attributes.n_angle, tomo_attributes.n_length))


