from ..rafttypes import *

import json
import h5py
import time
import numpy as np

class HDF5Reader:
    """
    Class for reading and preprocessing HDF5 data files for detectors.
    """

    def __init__(self, detector):
        """
        Initializes the HDF5Reader class with the provided detector type.

        Args:
            detector (str): The type of detector ('pco' or other).
        """
        self.detector = detector

    def _read_hdf5_file(self, filepath, hdf5path):
        """
        Reads HDF5 data from the specified file and path.

        Args:
            filepath (str): Path to the HDF5 file.
            hdf5path (str): Path within the HDF5 file to read the data from.

        Returns:
            ndarray: Data read from the HDF5 file.
        """
        start = time.time()
        data = h5py.File(filepath, "r")[hdf5path][:].astype(numpy.float32)
        elapsed = time.time() - start
        logger.info(f'Time for reading HDF5 file: {elapsed} seconds')
        return data

    def _preprocess_data(self, data):
        """
        Preprocesses the data based on its dimensions and the detector type.

        Args:
            data (ndarray): Data array to preprocess.

        Returns:
            ndarray: Preprocessed data array.
        """
        dim_data = len(data.shape)

        if self.detector == 'pco':
            if dim_data == 2:
                data[:11, :] = 1.0
            elif dim_data == 3:
                data[:, :11, :] = 1.0
            elif dim_data == 4:
                data[:, :, :11, :] = 1.0

        if dim_data == 4:
            #logger.debug(f"Data shape is 4, reducing to three!")
            data = data[:1, 0, :, :]
        elif dim_data == 2:
            #logger.debug(f"Data shape is 2, expanding to three!")
            data = numpy.expand_dims(data, axis=0)
        
        return data

    def read_data(self, filepath, hdf5path):
        """
        Reads and preprocesses the data from the HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
            hdf5path (str): Path within the HDF5 file to read the data from.

        Returns:
            ndarray: Preprocessed data array.
        """
        data = self._read_hdf5_file(filepath, hdf5path)
        return self._preprocess_data(data)

    def read_flat(self, filepath, hdf5path):
        """
        Reads and preprocesses the flat data from the HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
            hdf5path (str): Path within the HDF5 file to read the data from.

        Returns:
            ndarray: Preprocessed flat data array.
        """
        flat = self._read_hdf5_file(filepath, hdf5path)
        return self._preprocess_data(flat)

    def read_dark(self, filepath, hdf5path):
        """
        Reads and preprocesses the dark data from the HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
            hdf5path (str): Path within the HDF5 file to read the data from.

        Returns:
            ndarray: Preprocessed dark data array.
        """
        dark = self._read_hdf5_file(filepath, hdf5path)
        return self._preprocess_data(dark)


def read_tomo_flat_dark(dic):
    """
    Reads tomographic data, flat-field, and dark-field images from HDF5 files based on the input dictionary.

    This function attempts to read the tomographic data, flat-field, and dark-field images from the paths specified in the input dictionary.
    If the flat-field or dark-field images are not found, it uses default values (ones for flat-field and zeros for dark-field).

    Args:
        dic (dict): A dictionary containing the following keys:
            - 'Ipath' (str): The path to the input file. Default is ''.
            - 'Iname' (str): The name of the input file. Default is ''.
            - 'hdf5path' (str): The path inside the HDF5 file where the data is located (dataset). Default is 'scan/detector/data'.
            - 'Flatpath' (str): The path to the flat-field file. Default is the same as 'Ipath' + 'Iname'.
            - 'Darkpath' (str): The path to the dark-field file. Default is the same as 'Ipath' + 'Iname'.
            - 'Flathdf5path' (str): The path inside the HDF5 file where the flat-field data is located (dataset). Default is 'scan/detector/flats'.
            - 'Darkhdf5path' (str): The path inside the HDF5 file where the dark-field data is located (dataset). Default is 'scan/detector/darks'.
            - 'detector' (str): The detector type. Default is ''.

    Returns:
        tuple: A tuple containing:
            - tomogram (numpy.ndarray): The tomographic data.
            - flat_ (numpy.ndarray): The flat-field image.
            - dark_ (numpy.ndarray): The dark-field image.

    Raises:
        ValueError: If there is an error reading the tomographic data.

    Logs:
        Information about the paths being used, errors during reading, and timing for the read operations.
    """
    start = time.time()

    path = dic.get('input_path', '')
    name = dic.get('input_file_name', '')
    
    hdf5_path_data = dic.get('input_data_hdf5_path', 'scan/detector/data')

    flat_path = dic.get('flat_path', path + name)
    dark_path = dic.get('dark_path', path + name)

    hdf5_path_flat = dic.get('flat_path_hdf5_dataset', 'scan/detector/flats')
    hdf5_path_dark = dic.get('dark_path_hdf5_dataset', 'scan/detector/darks')

    # Input path and name
    filepath = path + name

    # Create tomogram instance to read HDF5 data
    tomogram_instance = HDF5Reader(dic.get('detector', ''))

    # Read raw data
    try:
        tomogram = tomogram_instance.read_data(filepath, hdf5_path_data)
    except Exception as e:
        logger.error(f'Error reading data in file {filepath}: {str(e)}')
        logger.error('Finishing run...')
        raise ValueError(f'Error reading data in file {filepath}: {str(e)}')

    try:
        flat = tomogram_instance.read_flat(flat_path, hdf5_path_flat)
    except Exception as e:
        logger.warning(f'No Flat-field was found in file {flat_path}: {str(e)}. Reconstruction continues with no Flat-field.')
        flat = numpy.ones((1, tomogram.shape[1], tomogram.shape[2]))

    try:
        dark = tomogram_instance.read_dark(dark_path, hdf5_path_dark)
    except Exception as e:
        logger.warning(f'No Dark-field was found in file {dark_path}: {str(e)}. Reconstruction continues with no Dark-field.')
        dark = numpy.zeros((1, tomogram.shape[1], tomogram.shape[2]))

    #flat_ = flat[0]
    #dark_ = dark[0]

    elapsed = time.time() - start
    logger.info(f'Time for Raft Read data, flat and dark: {elapsed} seconds')
    start = time.time()
    logger.info("Finished Raft Read data, flat and dark")

    return tomogram, flat, dark

def read_json(param):
   for j in range(len(param)):
      if param[j] == '-d':
         section = param[j+1]
      elif param[j] == '-f':
         name = param[j+1]

   jason = open(name)
   dic = json.load(jason)[section]

   return dic

def update_config_data(user_dictionary: dict) -> dict:
    """
    Creates a new dictionary with updated values based on the provided config data.

    Args:
        user_dictionary (dict): The original configuration data dictionary.

    Returns:
        dict: A new dictionary with updated values.
    """
    # Initialize updated_data with user_dictionary
    updated_data = dict(user_dictionary)

    # Handle flat_path and dark_path based on user input or default values
    input_full_path = updated_data["input_path"] + updated_data["input_file_name"]

    # Update flat_path
    updated_data["flat_path"] = updated_data.get("flat_path", input_full_path)

    # Update dark_path (defaults to flat_path if not specified)
    updated_data["dark_path"] = updated_data.get("dark_path", updated_data["flat_path"])

    # Check if specific file is being used
    if updated_data["input_file_name"] == "corrected_zoom_sinogram.h5":
        print("The data being used is the corrected_zoom_sinogram!")

    # Read data from the specified HDF5 file (flat_path)
    with h5py.File(updated_data["flat_path"], "r") as f:
        updated_data["z1[m]"] = f['snapshot/after/beamline-state/position/nano-station/z1/value'][()] * 1e-3
        updated_data["z2[m]"] = f['snapshot/after/beamline-state/position/nano-station/z2/value'][()] * 1e-3
        updated_data["z1+z2[m]"] = f['snapshot/after/beamline-state/position/nano-station/z1+z2/value'][()] * 1e-3
        updated_data["detector_pixel[m]"] = f['snapshot/after/beamline-state/position/detector/pixel-size-x/value'][()] * 1e-3
        updated_data["detector"] = f['scan/metadata/ad_default/PCO_Manufacturer'][0].astype(str).lower()

    # Update remaining paths
    updated_data["zoom_output_path"] = updated_data["output_path"] + "zoom_forward_reprojection.h5"
    updated_data["forward_zoom_file_path"] = updated_data["zoom_output_path"]
    updated_data["output_file_path"] = updated_data["output_path"] + "corrected_zoom_sinogram.h5"

    return updated_data


