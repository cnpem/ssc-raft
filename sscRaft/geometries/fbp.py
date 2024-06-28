# Authors: Giovanni L. Baraldi, Gilberto Martinez
from ..rafttypes import *
from ..processing.io import *
from .parallel.fbp import *


def fbp(tomogram, dic, angles = None, **kwargs):
    """Computes the reconstruction of a parallel beam tomogram using the 
    Backprojection Slice Theorem (BST) method.
    

    Args:
        tomogram (ndarray): Parallel beam projection tomogram. The axes are [slices, angles, lenght].
        dic (dict): Dictionary with the experiment info.

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z, y, x].

    * One or MultiGPUs. 
    * Calls function ``bstGPU()``
    * Calls function ``fbpGPU()``


    Dictionary parameters:

        * ``dic['gpu']`` (ndarray): List of gpus  [required]
        * ``dic['angles[rad]']`` (list): List of angles in radians [required]
        * ``dic['method']`` (str,optional):  [Default: 'RT']

             #. Options = (\'RT\',\'BST\')

        * ``dic['filter']`` (str,optional): Filter type [Default: \'lorentz\']

            #. Options = (\'none\',\'gaussian\',\'lorentz\',\'cosine\',\'rectangle\',\'hann\',\'hamming\',\'ramp\')
          
        * ``dic['beta/delta']`` (float,optional): Paganin by slices method ``beta/delta`` ratio [Default: 0.0 (no Paganin applied)]
        * ``dic['z2[m]']`` (float,optional): Sample-Detector distance in meters used on Paganin by slices method. [Default: 0.0 (no Paganin applied)]
        * ``dic['energy[eV]']`` (float,optional): beam energy in eV used on Paganin by slices method. [Default: 0.0 (no Paganin applied)]
        * ``dic['regularization']`` (float,optional): Regularization value for filter ( value >= 0 ) [Default: 1.0]
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [Default: 2]

    References:

        .. [1] Miqueles, X. E. and Koshev, N. and Helou, E. S. (2018). A Backprojection Slice Theorem for Tomographic Reconstruction. IEEE Transactions on Image Processing, 27(2), p. 894-906. DOI: https://doi.org/10.1109/TIP.2017.2766785.
    
    """
    required = ('gpu',)        
    optional = ( 'filter','offset','padding','regularization','beta/delta','blocksize','energy[eV]','z2[m]','method')
    default  = ('lorentz',       0,        2,             1.0,         0.0,          0,         0.0,    0.0,    'RT')
    
    dic = SetDictionary(dic,required,optional,default)

    dic['regularization'] = 1.0
    method                = dic['method']
    gpus                  = dic['gpu']

    if method == 'RT':
        try:
            angles = dic['angles[rad]']
        except:
            if angles is None:
                logger.error(f'Missing angles list!! Finishing run...') 
                raise ValueError(f'Missing angles list!!')

        output = fbpGPU( tomogram, angles, gpus, dic ) 

        return output
    
    if method == 'BST':

        angles = numpy.linspace(0.0, numpy.pi, tomogram.shape[-2], endpoint=False)

        output = bstGPU( tomogram, angles, gpus, dic ) 
    

        return output
