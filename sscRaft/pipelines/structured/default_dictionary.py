import subprocess
from multiprocessing import cpu_count

from sscRaft import __version__

def get_gpu_list() -> list:
    get_gpu = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l" # "nvidia-smi --list-gpus"
    gpu_list_process = subprocess.run(get_gpu, shell=True, capture_output=True, text=True)

    # Get the output and convert it to a list
    gpu_list = [gpu_index for gpu_index in range(int(gpu_list_process.stdout.strip()))]
    return gpu_list

default_dictionary ={
    "perform_data_reconstruction": True,
    "gpu": get_gpu_list(),
    "nprocesses": cpu_count(),
    "beam_energy": 22e3,
    "paganin_filter_method": 'none', # ['raft_by_frames', 'tomopy_by_frames', 'raft_by_slices', 'none']
    "paganin_filter_beta_delta": 0.0,
    "rings_method": "raft_titarenko", # ['raft_titarenko', 'raft_all_stripes', 'none']
    "rings_regularization_raft_ti": -1, 
    "rings_blocks_raft_ti": 1, 
    "rings_parameters": [10, 121, 21, 1],
    "projections": [0, -1],
    "rotation_axis_shift": [0, True],
    "recon_method": "raft_fdk",  # ['fbp_RT', 'fbp_BST', 'none']
    "slices": [0, -1],
    "recon_filter": "hamming",  # ['ramp', 'gaussian', 'hamming', 'hann', 'cosine', 'lorentz', 'rectangle']
    "padding": 2,
    "save_recon": True,
    "crop_circle_recon": True,
    "blocksize": 0,
    "ssc_raft_version": __version__
}

def update_with_defaults(input_dict, default_dict, missing_value=None) -> dict:
    """
    Update input_dict with default values from default_dict for any missing keys
    or keys with missing values.

    Args:
        input_dict (dict): The dictionary to update.
        default_dict (dict): The dictionary containing default key-value pairs.
        missing_value: The value that indicates a missing value in the input_dict.
                       Defaults to None, but can be set to other values as needed.

    Returns:
        dict: The updated dictionary with default values added for missing or empty keys.
    """
    # Copy the input dictionary to avoid modifying the original one
    updated_dict = input_dict.copy()
    
    # Iterate through the default dictionary
    for key, value in default_dict.items():
        # Check if the key is missing or its value is the missing_value
        if key not in updated_dict or updated_dict[key] == missing_value:
            updated_dict[key] = value
    updated_dict['energy[eV]'] = updated_dict['beam_energy']
    updated_dict['filter'] = updated_dict['recon_filter']
    updated_dict['crop circle'] = updated_dict['crop_circle_recon']
            
    return updated_dict


