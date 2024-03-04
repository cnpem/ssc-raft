# Authors: Giovanni L. Baraldi, Gilberto Martinez

from ...rafttypes import *
from ...processing.io import *
import numpy 

def bstGPU(tomogram, angles, gpus, dic):
           
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
    
    objsize        = nrays

    paganin        = dic['paganin regularization']
    regularization = dic['regularization']

    filter_type    = FilterNumber(dic['filter'])

    # logger.info(f'FBP Paganin regularization: {paganin}')

    pad            = dic['padding'] + 2

    # logger.info(f'Set BST pad value as {pad} x horizontal dimension ({padx}).')

    tomogram     = CNICE(tomogram) 
    tomogram_ptr = tomogram.ctypes.data_as(ctypes.c_void_p)

    obj          = numpy.zeros([nslices, objsize, objsize], dtype=numpy.float32)
    obj_ptr      = obj.ctypes.data_as(ctypes.c_void_p)

    angles       = numpy.array(angles)
    angles       = CNICE(angles) 
    angles_ptr   = angles.ctypes.data_as(ctypes.c_void_p) 

    libraft.getBSTMultiGPU(
            gpus_ptr, ctypes.c_int(ngpus), 
            obj_ptr, tomogram_ptr, angles_ptr, 
            ctypes.c_int(nrays), ctypes.c_int(nangles), ctypes.c_int(nslices), 
            ctypes.c_int(objsize), ctypes.c_int(pad), ctypes.c_float(regularization),
            ctypes.c_float(paganin), ctypes.c_int(filter_type))

    return obj


def bst(tomogram, dic, angles = None, **kwargs):
    """Computes the Reconstruction of a Parallel Tomogram using the 
    Backprojection Slice Theorem (BST) method.
    GPU function.

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        dic (dict): Dictionary with the experiment info.

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

    * One or MultiGPUs. 
    * Calls function ``bstGPU()``.

    Dictionary parameters:

        * ``dic['gpu']`` (ndarray): List of gpus for processing [required]
        * ``dic['angles[rad]']`` (list): list of angles in radians [required]
        * ``dic['filter']`` (str,optional): Filter type [default: \'lorentz\']

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')
          
        * ``dic['paganin regularization']`` (float,optional): Paganin regularization value ( value >= 0 ) [default: 0.0]
        * ``dic['regularization']`` (float,optional): Regularization value ( value >= 0 ) [default: 1.0]  
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 2]  

    """
    required = ('gpu',)        
    optional = ( 'filter','offset','padding','regularization','paganin regularization')
    default  = ('lorentz',       0,        2,             1.0,                     0.0)
    
    SetDictionary(dic,required,optional,default)

    dic['regularization'] = 1.0

    gpus  = dic['gpu']

    # objsize = dic['objsize']
    # if objsize % 32 != 0:
    #         objsize += 32-(objsize%32)
    #         logger.info(f'Reconstruction size not multiple of 32. Setting to: {objsize}')
    # dic.update({'objsize': objsize})

    angles = numpy.linspace(0.0, numpy.pi, tomogram.shape[-2], endpoint=False)
    # try:
    #     angles = dic['angles[rad]']
    # except:
    #     if angles is None:
    #         logger.error(f'Missing angles list!! Finishing run...') 
    #         raise ValueError(f'Missing angles list!!')

    output = bstGPU( tomogram, angles, gpus, dic ) 

    return output
