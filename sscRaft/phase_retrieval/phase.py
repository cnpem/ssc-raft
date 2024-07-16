from ..rafttypes import *
from ..processing.io import *

def phase_retrieval(frames, dic):

    required = ('gpu','detectorPixel[m]','z2[m]','energy[eV]','magn')
    optional = ('method', 'beta/delta','padding','blocksize')
    default  = ('paganin',          0.0,        2,          0)
    
    dic = SetDictionary(dic,required,optional,default)

    gpus     = dic['gpu']     
    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = frames.shape[-1]
    nslices  = frames.shape[-2]

    if len(frames.shape) == 2:
        nangles = 1
    else:
        nangles = frames.shape[0]

    beta_delta = dic['beta/delta']
    z2         = dic['z2[m]']
    energy     = dic['energy[eV]']
    magn       = dic['magn']
    pixel_det  = dic['detectorPixel[m]']

    padx, pady, padz  = dic['padding'],dic['padding'],0 # (padx, pady, padz)

    blocksize = dic['blocksize']

    if blocksize > ( nangles // ngpus ):
        logger.error(f'Blocksize is bigger than the number of angles ({nangles}) divided by the number of GPUs selected ({ngpus})!')
        raise ValueError(f'Blocksize is bigger than the number of angles ({nangles}) divided by the number of GPUs selected ({ngpus})!')

    methodname = dic['method']
    method     = PhaseMethodNumber(methodname)

    param_int     = [nrays, nslices, nangles, 
                     padx, pady, padz, method, blocksize]
    param_int     = numpy.array(param_int)
    param_int     = CNICE(param_int,numpy.int32)
    param_int_ptr = param_int.ctypes.data_as(ctypes.c_void_p)

    param_float     = [beta_delta,pixel_det,pixel_det,energy,z2,z2,magn,magn]
    param_float     = numpy.array(param_float)
    param_float     = CNICE(param_float,numpy.float32)
    param_float_ptr = param_float.ctypes.data_as(ctypes.c_void_p)
   
    frames = numpy.ascontiguousarray(frames.astype(numpy.float32))
    frames_ptr = frames.ctypes.data_as(ctypes.c_void_p)

    logger.info(f'Begin Phase Retrieval by {methodname}')

    libraft.getPhaseMultiGPU(gpus_ptr, ctypes.c_int(ngpus),
                            frames_ptr, param_float_ptr, param_int_ptr)                     

    logger.info(f'Finished Phase Retrieval')

    return frames

