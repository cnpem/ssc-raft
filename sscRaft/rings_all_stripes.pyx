# distutils: language = c

from cpython.ref cimport PyObject
cimport cython

cdef extern from "processing/rings_all_stripes.cuh":
    ctypedef struct TomoAttributes:
        int n_length
        int n_angle
        int n_slice
        int size_median
        int dim
        int large_stripes_size
        float background_ratio
        float drop_ratio
    void remove_all_stripes(float *tomo, TomoAttributes attribute, int ndevs, int batch)
    void remove_all_stripes_transposed(float *tomo, TomoAttributes attribute, int ndevs, int batch)

cdef TomoAttributes create_attributes(  int n_length, int n_angle, int n_slice,
                                        int size_median, int dim, int large_stripes_size,
                                        float background_ratio, float drop_ratio):
    cdef TomoAttributes tomo_attributes
    tomo_attributes.n_slice             = n_slice
    tomo_attributes.n_angle             = n_angle
    tomo_attributes.n_length            = n_length
    tomo_attributes.background_ratio    = background_ratio
    tomo_attributes.drop_ratio          = drop_ratio
    tomo_attributes.large_stripes_size  = large_stripes_size
    tomo_attributes.size_median         = size_median
    tomo_attributes.dim                 = dim
    return tomo_attributes

def _remove_all_stripes(float[:,:,:] tomo, int ndevs,   int batch,  int n_length, int n_angle, int n_slice,
                                                                int size_median, int dim, int large_stripes_size,
                                                                float background_ratio, float drop_ratio):
    cdef TomoAttributes tomo_attributes
    tomo_attributes.n_slice             = n_slice
    tomo_attributes.n_angle             = n_angle
    tomo_attributes.n_length            = n_length
    tomo_attributes.background_ratio    = background_ratio
    tomo_attributes.drop_ratio          = drop_ratio
    tomo_attributes.large_stripes_size  = large_stripes_size
    tomo_attributes.size_median         = size_median
    tomo_attributes.dim                 = dim

    remove_all_stripes(&tomo[0,0,0], tomo_attributes, ndevs, batch)

def _remove_all_stripes_multiaxis(float[:,:,:] tomo,
                                  int ndevs, int batch, int n_length, int n_angle, int n_slice,
                                  int size_median, int dim, int large_stripes_size,
                                  float background_ratio, float drop_ratio):
    cdef TomoAttributes tomo_attributes
    tomo_attributes.n_slice             = n_slice
    tomo_attributes.n_angle             = n_angle
    tomo_attributes.n_length            = n_length
    tomo_attributes.background_ratio    = background_ratio
    tomo_attributes.drop_ratio          = drop_ratio
    tomo_attributes.large_stripes_size  = large_stripes_size
    tomo_attributes.size_median         = size_median
    tomo_attributes.dim                 = dim

    remove_all_stripes_transposed(&tomo[0,0,0], tomo_attributes, ndevs, batch)
