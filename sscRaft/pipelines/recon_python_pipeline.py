from ..rafttypes import *
from .constructors import *
from ..geometries.methods import *
from ..io.saver import async_save_worker

import time

# Define the supported tomogram reconstruction methods
def set_fbp_method(method_name: str):
    return lambda tomogram, recon, dic: fbp(tomogram=tomogram, obj=recon, nstreams=2, dic={**dic, 'method': method_name})

# Define the supported tomogram reconstruction methods
def set_em_method(method_name: str):
    return lambda tomogram, recon, dic: em(tomogram=tomogram, obj=recon, dic={**dic, 'method': method_name})

# Define the supported tomogram reconstruction methods
recon_function_methods = {
    'fbp_RT': set_fbp_method('RT'),
    'fbp_BST': set_fbp_method('BST'),
    'tEMRT': set_em_method('tEMRT'),
    'tEMFQ': set_em_method('tEMFQ'),
    'eEMRT': set_em_method('eEMRT'), # Leave emission methods for the `ssc-xfct` package
    'none': dont_process
}

def multiple_recon_methods_wrapper(tomogram: numpy.ndarray, recon: numpy.ndarray, dic: dict) -> numpy.ndarray:
    """
    Wrapper function for multiple tomogram reconstruction methods.

    Args:
        tomogram (numpy.ndarray): The input tomogram.
        dic (dict): A dictionary containing all the parameters for the reconstruction.

    Returns:
        numpy.ndarray: The reconstructed tomogram.

    Raises:
        ValueError: If the specified reconstruction method is not implemented.
    """
    recon_method   = dic.get('method', 'fbp_BST')
    if recon_method in recon_function_methods:
        reconstruction = recon_function_methods[recon_method](tomogram=tomogram, recon=recon, dic=dic)
    else:
        raise ValueError(f"Unknown reconstruction method `{recon_method}` is not implemented!")

    return reconstruction

def reconstruction_methods(tomogram: numpy.ndarray, recon: numpy.ndarray, dic:dict) -> numpy.ndarray:
    """
    Perform tomogram reconstruction using multiple reconstruction methods.

    Args:
        tomogram (numpy.ndarray): The input tomogram.
        dic (dict): The dictionary containing reconstruction parameters.

    Returns:
        numpy.ndarray: The reconstructed tomogram.

    """
    start = time.time()

    # Perform the actual reconstruction
    recon = multiple_recon_methods_wrapper(tomogram, recon, dic)

    elapsed = time.time() - start
    logger.info(f'Finished data reconstruction! Total execution time: {elapsed:.2f} seconds')

    # Apply crop circle to reconstructed volume if specified
    if dic['crop_circle_recon']:
        recon = crop_circle_simple(recon, dic['detector_pixel[m]'],
                                   npixels=numpy.abs(dic['offset']))


    if dic.get('save_recon', False):
        save_name   = dic.get('reconstruct', 'Recon_' + dic['id'] + '_' + dic['input_name'])
        # Save the tomogram asynchronously if specified
        async_save_worker(recon, dic, save_name, 'save_recon')

    return recon

