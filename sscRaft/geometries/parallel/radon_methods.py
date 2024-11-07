from ...rafttypes import *

def radon_RT(phantom, angles, gpus, *args):
    """ Radon transform using Ray Tracing for a given input phantom 
    on a parallel beam geometry.
    
    Args:
        phantom (ndarray): digital squared input phantom. Axis are (slices, width, lenght) [required]
        angles (float list): list of angles in radians [required]
        gpus (int list): list of GPUs [required]
        args (float, optional): support domain at the x-axis [default: 1.0] 
             
    Returns:
        (ndarray): Radon transform 2D or 3D. Axis are (slices, angles, lenght)

    * MultiGPU function 
    """
    if not args:
        a = 1.0
    else:
        a = args[0] #numpy.sqrt(2.0)

    ngpus      = len(gpus)
    gpus       = numpy.array(gpus)
    gpus       = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpusptr    = gpus.ctypes.data_as(ctypes.c_void_p)

    if len(phantom.shape) == 2:
        nslices = 1
    else:
        nslices = phantom.shape[0]

    nrays       = phantom.shape[2]
    rays        = numpy.ceil( a * nrays ).astype(numpy.intc)

    _, img_sizey, img_sizex   = phantom.shape
    
    phantom    = CNICE(phantom)
    phantom_ptr = phantom.ctypes.data_as(ctypes.c_void_p)
      
    angles      = numpy.array(angles)
    nangles     = angles.shape[0]
    angles      = CNICE(angles)
    angles_ptr  = angles.ctypes.data_as(ctypes.c_void_p)  
    
    tomogram    = numpy.ones((nslices,nangles,rays), dtype=numpy.float32)
    tomogram *= -1
    tomogram      = CNICE(tomogram)
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)


    libraft.getRadonRTMultiGPU(gpusptr, ctypes.c_int(ngpus), 
        tomogram_ptr, phantom_ptr, angles_ptr, 
        ctypes.c_int(img_sizex), ctypes.c_int(img_sizey), 
        ctypes.c_int(rays), ctypes.c_int(nangles), ctypes.c_int(nslices), 
        ctypes.c_float(a), ctypes.c_float(a)) 
    
    # dt = (2.0*a)/(rays-1)
    # itmin = ( numpy.ceil( (-1 + a)/dt) ).astype(numpy.intc) 
    # itmax = ( numpy.ceil( ( 1 + a)/dt) ).astype(numpy.intc) 
    
    logger.info(f'Finished Radon RT method')
    return tomogram




