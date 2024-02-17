from ..rafttypes import *
import numpy as np
import gc
from time import time
from ctypes import c_float as float32
from ctypes import c_int as int32
from ctypes import c_void_p  as void_p
from ctypes import c_size_t as size_t

def phase_filters(tomo,dic):

    tomogram = numpy.copy(tomo)
    tomogram = np.swapaxes(tomogram,0,1)

    gpus = dic['gpu']     
    ngpus = len(gpus)

    gpus = np.array(gpus)
    gpus = np.ascontiguousarray(gpus.astype(np.intc))
    gpusptr = gpus.ctypes.data_as(void_p)

    nrays   = tomogram.shape[-1]
    nslices = tomogram.shape[-2]
    
    z1x       = dic['z1[m]']
    z2x       = dic['z2[m]']
    z1y       = z1x+0
    z2y       = z2x+0
    try:
       energy    = dic['energy[eV]']
    except:
        energy    = dic['energy[KeV]']


    alpha     = dic['phase filter regularization']
    pixel     = dic['detectorPixel[m]']

    padx = int(nrays // 2)
    pady = int(nrays // 2)
    try:
        pad  = dic['padding']
        padx = int(pad * padx)
        pady = int(pad * pady)
        # padx = power_of_2_padding(nrays,padx)
        logger.info(f'Set FDK pad value as ({padx},{pady}) = (padx,pady).')
    except:
        pad  = 1
        padx = int(pad * padx)
        pady = int(pad * pady)
        # padx = power_of_2_padding(nrays,padx)
        logger.info(f'Set default FDK pad value as ({padx},{pady}) = (padx,pady).')

    # Select beam geometry
    case = dic['beamgeometry']
    if (case == 'parallel'):
        logger.info('Parallel geometry selected')
        z1x, z1y = 0.0,0.0
    elif (case =='fanbeam'):
        logger.info('Fanbeam geometry selected')
        z1x = 0.0
    elif (case =='conebeam'):
        logger.info('Conebeam geometry selected')
    else:
        logger.error(f'Geometry does not exist! Select `parallel`, `fanbeam` or `conebem`.')
        sys.exit(1)

    if len(tomogram.shape) == 2:
        nangles = 1

    elif len(tomogram.shape) == 3:
        nangles = tomogram.shape[0]
    
    else:
        logger.error(f'Data has wrong shape = {tomogram.shape}. It needs to be 2D or 3D.')
        sys.exit(1)

    try:
        blocksize = dic['blocksize']
        
        if nangles == 1:
            blocksize = 1
        else:
            blocksize = dic['blocksize']
    except:
        blocksize = 32

        if nangles == 1:
            blocksize = 1
        else:
            blocksize = 32

    if blocksize > ( nangles // ngpus ):
        logger.error(f'Blocksize is bigger than the number of angles ({nangles}) divided by the number of GPUs selected ({ngpus})!')
        sys.exit(1)

    filtername = dic['phase filter method']
    filter     = PhaseFilterNumber(dic['phase filter'])
    
    if filter == 0:
        logger.warning(f'No phase filter selected! ')
        logger.warning(f'Finishing Run ...')
        sys.exit(0)
    else:
        logger.info(f'Phase filter method: {filtername}({filter})')

    float_param     = numpy.array([z1x, z1y, z2x, z2y, energy, alpha])
    float_param     = numpy.ascontiguousarray(float_param.astype(numpy.float32))
    float_paramsptr = float_param.ctypes.data_as(ctypes.c_void_p)

    int_param       = numpy.array([padx,pady,blocksize,filter])
    int_param       = numpy.ascontiguousarray(int_param.astype(numpy.int64))
    int_paramptr    = int_param.ctypes.data_as(ctypes.c_void_p)
   
    tomogram = np.ascontiguousarray(tomogram.astype(np.float32))
    tomogramptr = tomogram.ctypes.data_as(void_p)

    # libraft.phase_filters(tomogramptr, float_paramsptr, int_paramptr, int32(nrays), int32(nangles), int32(nslices), gpusptr, int32(ngpus))

    return np.swapaxes(tomogram,0,1)

