from ..rafttypes import *
import os
import time
import h5py
import asyncio
import numpy as np

from .._version import __version__
from .executor_manager import get_executor


# Define function to save the tomogram asynchronously
def async_save(tomogram: numpy.ndarray, dic: dict, save_prefix: str) -> None:
    """
    Asynchronously saves a tomogram to a file.

    Args:
        tomogram (numpy.ndarray): The tomogram to be saved.
        dic (dict): The user parameters input dictionary.
        save_prefix (str): The prefix to be used in the save file name.

    Returns:
        None
    """
    start = time.time()
    save_path = "".join([dic['output_path'], save_prefix, '_', dic['sample_id'], '_', dic['input_file_name']])
    logger.info(f"Saving File asynchronously to {save_path}...")
    hdf5_saver = HDF5Saver()
    hdf5_saver.save_hdf5_recon(save_path, tomogram, dic)  
    logger.info(f"File saved to {save_path} asynchronously.")
    elapsed = time.time() - start
    logger.info(f"Time: {elapsed:.2f} seconds")

# Define worker function to save the tomogram asynchronously
def async_save_worker(tomogram: numpy.ndarray, dic: dict, save_prefix: str, should_save_key: str) -> None:
    """
    Asynchronously saves the tomogram if specified.

    Args:
        tomogram (numpy.ndarray): The tomogram to be saved.
        dic (dict): The user parameters input dictionary.
        save_prefix (str): The prefix to be used for the saved file.
        should_save_key (str): The key in the dictionary indicating whether the tomogram should be saved.

    Returns:
        None
    """
    # Save the tomogram asynchronously if specified
    if dic[should_save_key]:
        executor = get_executor()
        executor.submit(async_save, tomogram, dic, save_prefix)


class HDF5Saver:
    """
    Class for saving data to HDF5 files with optional metadata.
    """

    def _save_hdf5(self, filepath, data_dict):
        """
        Saves data to an HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
            data_dict (dict): Dictionary of datasets to save in the HDF5 file.
        """
        with h5py.File(filepath, 'w') as file:
            for key, data in data_dict.items():
                file.create_dataset(key, data=data)

    def _save_metadata(self, file, dic, software, version):
        """
        Saves metadata to an HDF5 file.

        Args:
            file (h5py.File): HDF5 file object.
            dic (dict): Dictionary of metadata.
            software (str): Software name.
            version (str): Software version.
        """
        self.metadata_hdf5(file, dic, software, version)

    def metadata_hdf5(self, outputFileHDF5, dic, software, version):
        """
        Saves metadata from a dictionary to an HDF5 file.

        Args:
            outputFileHDF5 (h5py.File or str): HDF5 file object or file path.
            dic (dict): Dictionary containing metadata.
            software (str): Name of the software used.
            version (str): Version of the software.
        """
        dic['software'] = software
        dic['version'] = version

        if isinstance(outputFileHDF5, str):
            if os.path.exists(outputFileHDF5):
                hdf5 = h5py.File(outputFileHDF5, 'a')
            else:
                hdf5 = h5py.File(outputFileHDF5, 'w')
        else:
            hdf5 = outputFileHDF5

        for key, value in dic.items():
            h5_path = 'recon_parameters'
            hdf5.require_group(h5_path)

            if isinstance(value, numpy.ndarray):
                value = numpy.asarray(value)
                try:
                    hdf5[h5_path].create_dataset(key, data=value, shape=value.shape)
                except:
                    hdf5[h5_path][key] = value
            elif isinstance(value, (list, tuple)):
                value = str(value)
                try:
                    hdf5[h5_path].create_dataset(key, data=value, shape=())
                except:
                    hdf5[h5_path][key] = value
            else:
                try:
                    hdf5[h5_path].create_dataset(key, data=value, shape=())
                except:
                    hdf5[h5_path][key] = value

        if isinstance(outputFileHDF5, str):
            hdf5.close()

    def save_hdf5_phase_retrieval(self, filepath, tomogram, dic):
        """
        Saves phase retrieval data to an HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
            tomogram (ndarray): Complex tomogram data.
            dic (dict): Metadata dictionary.
        """
        data_dict = {
            "absorption": tomogram[0],
            "phase": tomogram[1]
        }
        self._save_hdf5(filepath, data_dict)
        self._save_metadata(h5py.File(filepath, 'a'), dic, 'sscRaftApps', __version__)

    def save_hdf5_complex(self, filepath, tomogram):
        """
        Saves complex tomogram data to an HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
            tomogram (ndarray): Complex tomogram data.
        """
        data_dict = {
            "absorption": tomogram.real,
            "phase": tomogram.imag
        }
        self._save_hdf5(filepath, data_dict)

    def save_hdf5_recon(self, filepath, recon, dic):
        """
        Saves reconstruction data to an HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
            recon (ndarray): Reconstruction data.
            dic (dict): Metadata dictionary.
        """
        data_dict = {
            "data": recon
        }
        self._save_hdf5(filepath, data_dict)
        self._save_metadata(h5py.File(filepath, 'a'), dic, 'sscRaftApps', __version__)

    def save_meta(self, filepath, data, dic):
        """
        Saves data to an HDF5 file with optional metadata.

        Args:
            filepath (str): Path to the HDF5 file.
            data (ndarray): Data to save.
            dic (dict): Metadata dictionary.
        """
        if numpy.issubdtype(data.dtype, numpy.complexfloating):
            self.save_hdf5_phaseRet(filepath, data, dic)
        else:
            self.save_hdf5_recon(filepath, data, dic)

    def save(self, filepath, data):
        """
        Saves data to an HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
            data (ndarray): Data to save.
        """
        if numpy.issubdtype(data.dtype, numpy.complexfloating):
            self.save_hdf5_complex(filepath, data)
        else:
            self._save_hdf5(filepath, {"data": data})
