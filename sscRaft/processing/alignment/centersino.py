from ...rafttypes import *
import numpy as np

from ctypes import c_int as int32
from ctypes import c_void_p  as void_p


def Centersino(frame0, frame1, flat, dark):
        """ Find the offset of a 180 tomogam to correctly align it, computed by cross correlation.
        It does the application of Flatfield :math:`I_{0}` and darkfield :math:`d_{0}` 
        on the intensity data :math:`I` inside function

        .. math:: -\log(\frac{I - d_{0}}{I_{0}})

        Args:
            frame0 (ndarray): First frame of intensity data obtained by detector
            frame1 (ndarray): Last frame of intensity data obtained by detector
            flat (ndarray): Flat
            dark (ndarray): Dark

        Returns:
            int: offset
        """        

        nrays = frame0.shape[-1]
        nslices = frame0.shape[-2]

        frame0 = np.ascontiguousarray(frame0.astype(np.float32))
        frame0ptr = frame0.ctypes.data_as(void_p)

        frame1 = np.ascontiguousarray(frame1.astype(np.float32))
        frame1ptr = frame1.ctypes.data_as(void_p)

        dark = np.ascontiguousarray(dark.astype(np.float32))
        darkptr = dark.ctypes.data_as(void_p)

        flat = np.ascontiguousarray(flat.astype(np.float32))
        flatptr = flat.ctypes.data_as(void_p)

        offset = libraft.findcentersino(frame0ptr, frame1ptr, darkptr, flatptr, int32(nrays), int32(nslices))
        
        return int(offset)
