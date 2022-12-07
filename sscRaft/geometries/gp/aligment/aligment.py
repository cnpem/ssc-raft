from ....rafttypes import *
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
        gpus = np.ascontiguousarray(gpus.astype(np.int32))
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


def _worker_360to180_(params, start, end, gpu, process):

        output = params[0]
        data   = params[1]
        offset = params[6]

        logger.info(f'Applying Rings: begin process {process} on gpu {gpu}')

        output[start:end,:,:] = Tomo360To180GPU( data[start:end, :, :], offset, gpu )

        logger.info(f'Applying Rings: end process {process} on gpu {gpu}')
    

def _build_360to180_(params):
 
        nslices = params[2]
        gpus    = params[5]
        ngpus = len(gpus)

        b = int( numpy.ceil( nslices/ngpus )  ) 

        processes = []
        for process in range( ngpus ):
                begin_ = process*b
                end_   = min( (process+1)*b, nslices )

                p = multiprocessing.Process(target=_worker_360to180_, args=(params, begin_, end_, gpus[process], process))
                processes.append(p)
    
        for p in processes:
                p.start()

        for p in processes:
                p.join()


def tomo360to180_gpublock( tomogram, offset, gpus ):

        nslices     = tomogram.shape[0]
        nangles     = tomogram.shape[1]
        nrays       = tomogram.shape[2]

        name = str( uuid.uuid4())

        try:
                sa.delete(name)
        except:
                pass

        output  = sa.create(name,[nslices, nangles // 2, 2 * nrays], dtype=np.float32)

        _params_ = ( output, tomogram, nslices, nangles, nrays, gpus, offset)

        _build_360to180_( _params_ )

        sa.delete(name)

        return output


def Tomo360To180_threads(tomogram, offset, gpus = [0], **kwargs):

        if len(gpus) == 1:
                gpu = gpus[0]
                output = Tomo360To180GPU( tomogram, offset, gpu )
        else:
                output = tomo360to180_gpublock( tomogram, offset, gpus ) 

        return output


def Tomo360To180(tomogram, offset, gpus = [0], **kwargs):

        if len(gpus) == 1:
                gpu = gpus[0]
                output = Tomo360To180GPU( tomogram, offset, gpu )
        else:
                output = Tomo360To180MultiGPU( tomogram, offset, gpus ) 

        return output