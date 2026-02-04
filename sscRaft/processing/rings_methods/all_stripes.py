import numpy as np
from ...rings_all_stripes import _remove_all_stripes, _remove_all_stripes_multiaxis
from .class_rings import TomoAttributes_


def set_tomo_attributes(size, settings):
    (n_slice, n_angle, n_length)    = size
    size_median                     = settings['size_median'] + 1 if settings['size_median'] % 2 == 0 else settings['size_median']
    dim                             = settings['dim']
    large_stripes_size              = settings['large_stripes_size'] + 1 if settings['large_stripes_size'] % 2 == 0 else settings['large_stripes_size']
    background_ratio                = settings['background_ratio']
    drop_ratio                      = settings['drop_ratio']

    return TomoAttributes_( n_length, n_angle, n_slice,
                            size_median, dim, large_stripes_size,
                            background_ratio, drop_ratio)

def rings_remove_all_stripes(tomo, ngpus, settings, batch = 256):

    tomo_attributes = set_tomo_attributes(tomo.shape, settings)
    if tomo.dtype != np.float32:
        tomo = tomo.astype('float32')
    if not tomo.data.c_contiguous:
        tomo = np.ascontiguousarray(tomo)

    remove_all_stripes(tomo, ngpus, batch, tomo_attributes)

    return np.reshape(tomo, (tomo_attributes.n_slice, tomo_attributes.n_angle, tomo_attributes.n_length))

def remove_all_stripes(tomo, ngpus: int, batch:int, tomo_attributes: TomoAttributes_):
    _remove_all_stripes(tomo, ngpus, batch, tomo_attributes.n_length, tomo_attributes.n_angle, tomo_attributes.n_slice,
                                            tomo_attributes.size_median, tomo_attributes.dim, tomo_attributes.large_stripes_size,
                                            tomo_attributes.background_ratio, tomo_attributes.drop_ratio)

def rings_remove_all_stripes_multiaxis(tomo, ngpus, settings, batch = 256):
    tomo_attributes = set_tomo_attributes(tomo.shape, settings)
    if tomo.dtype != np.float32:
        tomo = tomo.astype('float32')
    if not tomo.data.c_contiguous:
        tomo = np.ascontiguousarray(tomo)

    _remove_all_stripes_multiaxis(tomo, ngpus, batch,
                        tomo_attributes.n_length, tomo_attributes.n_angle, tomo_attributes.n_slice,
                        tomo_attributes.size_median, tomo_attributes.dim,
                        tomo_attributes.large_stripes_size, tomo_attributes.background_ratio,
                        tomo_attributes.drop_ratio)

    return np.reshape(tomo, (tomo_attributes.n_slice, tomo_attributes.n_angle, tomo_attributes.n_length))


