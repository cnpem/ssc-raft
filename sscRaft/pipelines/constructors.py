from ..rafttypes import *
import sys
import time
import numpy as np
# from .io.saver import async_save_worker, async_save_worker_phase

# Generic processing function
def apply_filter_constructor(tomogram: numpy.ndarray, 
                             dic: dict, 
                             process_fn, 
                             process_name: str, 
                             save_prefix: str, 
                             should_save_key: str) -> numpy.ndarray:
    """
    Apply a filter constructor to a tomogram.

    Args:
        tomogram (numpy.ndarray): The input tomogram.
        dic (dict): The dictionary containing all filter parameters.
        process_fn: The filter process function to apply to the tomogram.
        process_name (str): The name of the filter process.
        save_prefix (str): The prefix for saving the tomogram.
        should_save_key (str): The key indicating whether to save the tomogram.

    Returns:
        numpy.ndarray: The filtered 
    """
    # Start the timing for the process
    start = time.time()
    
    # Apply the specific filter process
    logger.info(f"Applying {process_name}...")
    tomogram = process_fn(tomogram, dic)
    
    # Calculate and log the elapsed time
    elapsed = time.time() - start
    logger.info(f'Finished {process_name}! Time: {elapsed:.2f} seconds')
    
    # Save the tomogram asynchronously if specified
    # async_save_worker(tomogram, dic, save_prefix, should_save_key) 
    
    return tomogram

def dont_process(tomogram: numpy.ndarray, dic: dict) -> numpy.ndarray:
    """
    Returns the input tomogram unchanged.

    This function is a placeholder that performs no processing on the input tomogram. It simply 
    returns the input data as-is.

    Args:
        tomogram (numpy.ndarray): 
            The input tomogram to be returned unchanged.
        dic (dict): 
            A dictionary of parameters, which are ignored in this function.

    Returns:
        numpy.ndarray: The input tomogram, unchanged.
    """
    return tomogram

def process_tomogram_volume(tomogram: numpy.ndarray, 
                            dic: dict,
                            process_fn: callable) -> numpy.ndarray:
    """
    Processes a tomogram using a provided processing function.

    Args:
        tomogram (numpy.ndarray): 
            The input tomogram
        dic (dict): 
            A dictionary containing parameters required by the processing function.
        process_fn (callable): 
            A function that takes a NumPy array and a dictionary as input and returns a processed 
            NumPy array.

    Returns:
        Processed (numpy.ndarray): 
            The processed tomogram according to process_fn
    """
    return process_fn(tomogram, recon, dic)

