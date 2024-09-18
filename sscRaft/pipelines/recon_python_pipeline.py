from ..rafttypes import *
from .constructors import *
from ..geometries.methods import *
from ..io.saver import async_save_worker

# Define the supported tomogram reconstruction methods
def set_fbp_method(method_name: str):
    return lambda tomogram, dic: fbp(tomogram, dic={**dic, 'method': method_name})

# Define the supported tomogram reconstruction methods
def set_em_method(method_name: str):
    return lambda tomogram, dic: em(tomogram, dic={**dic, 'method': method_name})

# Define the supported tomogram reconstruction methods
recon_function_methods = {
    'fbp_RT': set_fbp_method('RT'),
    'fbp_BST': set_fbp_method('BST'),
    'tEMRT': set_em_method('tEMRT'),
    'tEMFQ': set_em_method('tEMFQ'),
    'eEMRT': set_em_method('eEMRT'), # Leave emission methods for the `ssc-xfct` package
    'none': dont_process
}

def multiple_recon_methods_wrapper(tomogram: numpy.ndarray, dic: dict) -> numpy.ndarray:
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
    reconstruction = dic['recon array']

    recon_method   = dic.get('recon_method', 'fbp_BST')
    if recon_method in recon_function_methods:
        reconstruction = recon_function_methods[recon_method](tomogram=tomogram, dic=dic, obj=reconstruction)
    else:
        raise ValueError(f"Unknown reconstruction method `{recon_method}` is not implemented!")

    return reconstruction

def reconstruction_methods(tomogram: numpy.ndarray, dic:dict) -> numpy.ndarray:
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
    recon = multiple_recon_methods_wrapper(tomogram, dic)

    elapsed = time.time() - start
    logger.info(f'Finished data reconstruction! Total execution time: {str(elapsed)} seconds')

    # Apply crop circle to reconstructed volume if specified
    if dic['crop_circle_recon']:
        recon = crop_circle_simple(recon, dic['detector_pixel[m]'],
                                   npixels=numpy.abs(dic['rotation_axis_shift'][0]))

    elapsed = time.time() - start

    # Save the tomogram asynchronously if specified
    async_save_worker(recon, dic, 'recon', 'save_recon')

    return recon

