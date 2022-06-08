from ..rafttypes import *
import numpy as np
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p


def Find360Offset(tomo, gpu = [0]):
        """Computes the offset of the 360 tomogram fullview to merge the two aquisitions.
        It is obtained by a correlation method of the two sides of the aquisition.

        Args:
        tomo (ndarray): Tomogram. The axes are [angles, slices, X].
        gpu (list of ints): The gpus to be used. Defaults to [0]

        Returns:
        (int): Offset
        """
        
        sizex = tomo.shape[-1]
        sizey = tomo.shape[-2]

        if len(tomo.shape)==2:
                sizez = 1
        else:
                sizez = tomo.shape[0]

        tomo = np.ascontiguousarray(tomo.astype(np.float32))
        tomoptr = tomo.ctypes.data_as(void_p)

        offset = libraft.CPUTomo360_PhaseCorrelation(int32(gpu), tomoptr, int32(sizex), int32(sizey), int32(sizez))

        return int(offset)

def Tomo360To180(tomo, offset, gpu=[0]):
        """Computes the merge of the 360 tomogram fullview to a 180 tomogram.
        The merge considers the input offset and gradually adjust the gray values on the offset interval

        Args:
        tomo (ndarray): Tomogram. The axes are [angles, slices, X].
        offset (int): the offset value for the merge.
        gpu (list of ints): The gpus to be used. Defaults to [0]

        Returns:
        (int): Offset
        """
        
        sizex = tomo.shape[-1]
        sizey = tomo.shape[-2]

        if len(tomo.shape)==2:
                sizez = 1
        else:
                sizez = tomo.shape[0]

        tomo = np.ascontiguousarray(tomo.astype(np.float32))
        tomoptr = tomo.ctypes.data_as(void_p)

        libraft.CPUTomo360_To_180(int32(gpu), tomoptr, int32(sizex), int32(sizey), int32(sizez), int32(offset))
        
        return tomo.reshape((sizez,sizey//2,sizex*2))
