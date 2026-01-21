from ...rafttypes import *
import numpy
import ctypes


def ExcentricOverlapInterpolation(name):
    if name.lower() == 'none':
        return 10
    elif name.lower() == 'linear':
        return 0
    elif name.lower() == 'gaussian':
        return 1
    elif name.lower() == 'cosine':
        return 2
    elif name.lower() == 'sigmoid':
        return 3
    else:
        return 10
    
def getOffsetExcentricTomo(tomogram, gpus = [0]):
        """Computes the offset of the 360 tomogram fullview to merge the two aquisitions.
        It is obtained by a correlation method of the two sides of the aquisition.

        Args:
            tomo (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
            gpu (list of ints): The gpus to be used. Defaults to [0]

        Returns:
            (int): Offset
        """
        gpu = gpus[0]

        nrays = tomogram.shape[-1]
        nangles = tomogram.shape[-2]

        if len(tomogram.shape) == 2:
            nslices = 1
        else:
            nslices = tomogram.shape[0]

        tomogram     = CNICE(tomogram) 
        tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

        offset = libraft.getOffsetExcentricTomoGPU(ctypes.c_int(gpu), tomogram_ptr, 
                                               ctypes.c_int(nrays), ctypes.c_int(nangles), ctypes.c_int(nslices))
    
        return int(offset)

    
def stitchExcentricTomo(tomogram: np.ndarray, offset: int, gpus: list = [0], method: str = 'none') -> np.ndarray:
    """Computes the transformation of a 360 degree measured tomogram fullview to a 180 degree measured tomogram.
    The transformation considers the input offset and gradually adjust the gray values on the offset interval

    Args:
        tomogram (ndarray): A 360 degree measured tomogram. The axes are [slices, angles, nrays] 
        offset (int): The offset value for the transformation, with sign
        gpus (list of ints, optional): The gpus to be used. [default: [0]]
        method (str, optional): Intensity interpolation on overlapping area. Options: \'none\', \'linear\', \'gaussian\', \'cosine\', \'sigmoid\' [default: \'none\']

    Returns:
        (ndarray): A 180 degree tomogram. The axes are [slices, angles, nrays]
    """

    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = tomogram.shape[-1]
    nangles  = tomogram.shape[-2]
    
    if len(tomogram.shape) == 2:
        nslices = 1
    else:
        nslices = tomogram.shape[0]
    
    if nangles % 2 != 0:
        tomogram     = CNICE(tomogram[:,:-1,:])
        tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)
    else:
        tomogram     = CNICE(tomogram)
        tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    Interpolation_method = ExcentricOverlapInterpolation(method)
    
    libraft.getExcentricTomoMultiGPU(
        gpus_ptr, 
        ctypes.c_int(ngpus), 
        tomogram_ptr, 
        ctypes.c_int(nrays), 
        ctypes.c_int(nangles), 
        ctypes.c_int(nslices), 
        ctypes.c_int(offset), 
        ctypes.c_int(Interpolation_method)
    )

    return tomogram.reshape((nslices, nangles // 2, nrays * 2))
    
