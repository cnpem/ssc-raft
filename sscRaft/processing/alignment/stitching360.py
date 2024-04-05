from ...rafttypes import *

def getOffsetStitching360(tomogram, gpus = [0]):
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

        offset = libraft.getOffsetStitch360GPU(ctypes.c_int(gpu), tomogram_ptr, 
                                               ctypes.c_int(nrays), ctypes.c_int(nangles), ctypes.c_int(nslices))
    
        return int(offset)


def stitch360To180GPU(tomogram, offset, gpus = [0]):
    """Computes the merge of the 360 tomogram fullview to a 180 tomogram.
    The merge considers the input offset and gradually adjust the gray values on the offset interval

    Args:
        tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        offset (int): the offset value for the merge.
        gpus (list of ints): The gpus to be used. Defaults to [0]

    Returns:
        (ndarray): 180 degrees tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
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

    tomogram     = CNICE(tomogram)
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    libraft.stitch360To180MultiGPU(gpus_ptr, ctypes.c_int(ngpus), 
                                tomogram_ptr, 
                                ctypes.c_int(nrays), ctypes.c_int(nangles), ctypes.c_int(nslices), 
                                ctypes.c_int(offset))

    return tomogram.reshape((nslices, nangles // 2, nrays * 2))

def stitch360To180(tomogram, offset, gpus = [0]):

    return stitch360To180GPU( tomogram, offset, gpus ) 
                
