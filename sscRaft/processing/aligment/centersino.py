from ...rafttypes import *
import numpy as np
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t
import uuid
import SharedArray as sa

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

        start = time()

        offset = libraft.findcentersino(frame0ptr, frame1ptr, darkptr, flatptr, int32(nrays), int32(nslices))

        elapsed = time() - start
        
        return int(offset)

def Centersino16(frame0, frame1, flat, dark):
        """ Uint16 function to find the offset of a 180 tomogam to correctly align it, computed by cross correlation. 
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

        frame0 = CNICE(frame0,np.uint16)
        frame0ptr = frame0.ctypes.data_as(void_p)

        frame1 = CNICE(frame1,np.uint16)
        frame1ptr = frame1.ctypes.data_as(void_p)

        dark = CNICE(dark,np.uint16)
        darkptr = dark.ctypes.data_as(void_p)

        flat = CNICE(flat,np.uint16)
        flatptr = flat.ctypes.data_as(void_p)

        start = time()

        offset = libraft.findcentersino16(frame0ptr, frame1ptr, darkptr, flatptr, int32(nrays), int32(nslices))

        elapsed = time() - start
       
        return int(offset)



def Find360Offset(tomogram, gpus = [0]):
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

        tomogram = np.ascontiguousarray(tomogram.astype(np.float32))
        tomogramptr = tomogram.ctypes.data_as(void_p)

        start = time()

        offset = libraft.ComputeTomo360Offsetgpu(int32(gpu), tomogramptr, int32(nrays), int32(nangles), int32(nslices))
       
        elapsed = time() - start

        return int(offset)


def Find360Offset16(data, flat, dark, gpus = [0]):
        """Uint16 function to compute the offset of the 360 tomogram fullview to merge the two aquisitions, for more than one flat.
        It is obtained by a correlation method of the two sides of the aquisition.
        
        It does the application of Flatfield :math:`I_{0}` and darkfield :math:`d_{0}` 
        on the intensity data :math:`I` inside function

        .. math:: -\log(\frac{I - d_{0}}{I_{0}})

        Args:
        data (ndarray): Raw intensity data of detector (3D or 2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        gpu (list of ints): The gpus to be used. Defaults to [0]

        Returns:
        (int): Offset
        """
        gpu = gpus[0]

        nrays = data.shape[-1]
        nangles = data.shape[0]

        if len(data.shape) == 2:
                nslices = 1
        else:
                nslices = data.shape[-1]
        
        data = CNICE(data,np.uint16)
        dataptr = data.ctypes.data_as(void_p)

        flat = CNICE(flat,np.uint16)
        flatptr = flat.ctypes.data_as(void_p)

        dark = CNICE(dark,np.uint16)
        darkptr = dark.ctypes.data_as(void_p)

        numflats = int32( flat.shape[0] if len(flat.shape)==3 else 1 )

        start = time()

        offset = libraft.ComputeTomo360Offset16(int32(gpu), dataptr, flatptr, darkptr, int32(nrays), int32(nslices), int32(nangles), numflats)

        elapsed = time() - start

        return int(offset)

def Tomo360To180MultiGPU(tomogram, offset, gpu = [0]):
        """Computes the merge of the 360 tomogram fullview to a 180 tomogram.
        The merge considers the input offset and gradually adjust the gray values on the offset interval

        Args:
        tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        offset (int): the offset value for the merge.
        gpus (list of ints): The gpus to be used. Defaults to [0]

        Returns:
        (ndarray): 180 degrees tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        """     
        ngpus = len(gpu)

        gpus = numpy.array(gpu)
        gpus = np.ascontiguousarray(gpus.astype(np.intc))
        gpusptr = gpus.ctypes.data_as(void_p)  

        nrays = tomogram.shape[-1]
        nangles = tomogram.shape[-2]
        
        if len(tomogram.shape) == 2:
                nslices = 1
        else:
                nslices = tomogram.shape[0]

        tomogram = np.ascontiguousarray(tomogram.astype(np.float32))
        tomogramptr = tomogram.ctypes.data_as(void_p)

        start = time()

        libraft.Tomo360To180block(gpusptr, int32(ngpus), tomogramptr, int32(nrays), int32(nangles), int32(nslices), int32(offset))

        elapsed = time() - start

        return tomogram.reshape((nslices, nangles // 2, nrays * 2))

def Tomo360To180GPU(tomogram, offset, gpu = 0):
        """Computes the merge of the 360 tomogram fullview to a 180 tomogram.
        The merge considers the input offset and gradually adjust the gray values on the offset interval

        Args:
        tomogram (ndarray): Tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        offset (int): the offset value for the merge.
        gpus (list of ints): The gpus to be used. Defaults to [0]

        Returns:
        (ndarray): 180 degrees tomogram (3D) or sinogram (2D). The axes are [slices, angles, nrays] (3D) or [angles, nrays] (2D).
        """     
        
        nrays = tomogram.shape[-1]
        nangles = tomogram.shape[-2]
        
        if len(tomogram.shape) == 2:
                nslices = 1
        else:
                nslices = tomogram.shape[0]

        tomogram = np.ascontiguousarray(tomogram.astype(np.float32))
        tomogramptr = tomogram.ctypes.data_as(void_p)

        start = time()

        libraft.Tomo360To180gpu(int32(gpu), tomogramptr, int32(nrays), int32(nangles), int32(nslices), int32(offset))

        elapsed = time() - start

        return tomogram.reshape((nslices, nangles // 2, nrays * 2))


def Tomo360To180(tomogram, offset, gpus = [0], **kwargs):

        if len(gpus) == 1:
                gpu = gpus[0]
                output = Tomo360To180GPU( tomogram, offset, gpu )
        else:
                output = Tomo360To180MultiGPU( tomogram, offset, gpus ) 
                
        # Garbage Collector
        # lists are cleared whenever a full collection or
        # collection of the highest generation (2) is run
        # collected = gc.collect() # or gc.collect(2)
        # logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')

        return output