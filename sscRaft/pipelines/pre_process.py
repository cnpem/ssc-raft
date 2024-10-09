from ..rafttypes import *

# from ..io.saver import async_save_worker

from ..processing.background_correction import *
from ..processing.alignments.rotationaxis import *
from ..processing.alignments.stitching360 import *

from ..processing.opt import transpose

import time

def apply_log(tomogram: numpy.ndarray, dic: dict) -> numpy.ndarray:
    start = time.time()
    logger.info('Start applying -log(tomogram)...')
    tomogram = -numpy.log(tomogram, tomogram) # calculate inplace the -log(tomogram)
    logger.info(f'Finished applying -log(tomogram). Time elapsed: {time.time() - start}s')

    return tomogram

def calculate_transposed_volumes(tomogram: numpy.ndarray, flat: numpy.ndarray, dark: numpy.ndarray) -> tuple:
    start = time.time()
    logger.info('Calculating the transpose volumes of tomogram, flat-field and dark-field...')
    flat = transpose(flat)
    dark = transpose(dark)
    tomogram = transpose(tomogram)
    logger.info(f'Finished calculating transpose volumes. Time elapsed: {str(time.time() - start)}s')

    return tomogram, flat, dark

def update_paganin_regularization(dic: dict) -> dict:
    # Check if 'paganin_method' is in the dictionary
    if 'paganin_method' not in dic:
        raise KeyError("The key 'paganin_method' is missing from the dictionary.")
    
    # Update regularization value if the method is not 'paganin_by_slices'
    if dic['paganin_method'] != 'paganin_by_slices':
        dic['beta/delta'] = 0.0
    
    return dic

def fix_stitching(tomogram: numpy.ndarray, recon: numpy.ndarray, dic: dict) -> numpy.ndarray:
    start = time.time()

    deviation    = dic.get('stitching overlap', 0)
    gpus         = dic.get('gpu',[0])

    tomogram = stitch360To180(tomogram, deviation, gpus=gpus)

    # Calculate and log the elapsed time
    elapsed = time.time() - start
    logger.info(f'Time for Stitching: {elapsed:.2f} seconds')

    logger.info("Finished Stitching")

    # Asyncronously save the processed Stitching tomogram volume
    # async_save_worker(tomogram, dic, 'raft_rotation_axis', 'save_rotation_axis') 

    return tomogram

def fix_rotation_axis(tomogram: numpy.ndarray, recon: numpy.ndarray, dic: dict) -> numpy.ndarray:
    start = time.time()

    deviation = dic.get('axis offset', 0)

    # Apply rotation axis correction

    tomogram = correct_rotation_axis_cropped(tomogram, deviation)

    # Calculate and log the elapsed time
    elapsed = time.time() - start
    logger.info(f'Time for Raft Rotation Axis: {elapsed:.2f} seconds')
    
    logger.info("Finished Raft Rotation Axis")

    # Asyncronously save the processed rotation axis tomogram volume
    # async_save_worker(tomogram, dic, 'raft_rotation_axis', 'save_rotation_axis') 

    return tomogram

def angle_verification(dic: dict, tomogram: numpy.ndarray) -> numpy.ndarray:
    file_path = dic['input_path']
    file_name = dic['input_name']
    abs_file_path = "".join([file_path, file_name])

    tomogram_shape_angles = tomogram.shape[1]

    # Try to read the angles from the HDF5 file
    try:
        with h5py.File(abs_file_path, 'r') as h5f:
            # Attempt to read the angles from the dataset
            try:
                angles = h5f['exchange/theta'][:].astype(numpy.float32)
            except KeyError:
                logger.error(f"Dataset 'exchange/theta' not found in {abs_file_path}.")
                raise

        # Convert angles to radians and store the last angle from the vector
        angles = numpy.asarray(numpy.deg2rad(angles))
        angles = abs(min(angles)) + angles
        last_angle_from_vector = angles[-1]

        # Ensure angles array matches the tomogram shape
        if len(angles) != tomogram_shape_angles:
            logger.warning(f"Number of angles ({len(angles)}) in HDF5 file does not match the tomogram ({tomogram_shape_angles}) number of projections.")
            logger.info("Creating a new list of angles based on the last angle from the HDF5 angle vector.")
            angles = numpy.linspace(0.0, last_angle_from_vector, tomogram_shape_angles, endpoint=False)

    except (IOError, KeyError) as e:

        logger.error(f"Error reading angles from HDF5 file {abs_file_path}: {str(e)}.")
    
    return angles

def find_rotation_axis_auto(dic:dict, angles: numpy.ndarray, data: numpy.ndarray, flat: numpy.ndarray, dark: numpy.ndarray) -> int:
    deviation = dic.get('axis offset', 0)  
    deviation = dic.get('axis offset auto', False)

    try:
        # Find index of projection at 180 degrees to compute the rotation axis deviation
        dangle = angles[1] - angles[0]
        iangle = ( numpy.pi / dangle ).astype(int)
        frame0 = data[:,0,:]
        frame1 = data[:,iangle-1,:]

    except:
        message = (
            'Number of angles less than 180 degrees.\n'
            'Trying to find rotation axis deviation with first and last frames.\n'
            'Careful! The value can be skewed.'
        )
        logger.warning(message)

        frame0 = data[:,0,:]
        frame1 = data[:,-1,:]

    if dic.get('axis offset auto', True):
        deviation = Centersino(frame0, frame1, flat, dark)  
        logger.info(f'Rotation axis offset computed: {deviation} pixels')
    else:
        logger.warning(f'Rotation axis deviation value of {deviation} pixels provided by the user.')

    return deviation

def find_stitching_auto(dic: dict, tomogram: numpy.ndarray) -> int:
    gpus  = dic.get('gpu',[0])

    deviation = getOffsetStitching360(tomogram[:2, :, :], gpus=gpus)

    logger.info(f'Stitching overlap computed: {deviation}')

    return deviation


def select_slices_to_reconstruct(tomogram: numpy.ndarray, flat: numpy.ndarray, dark: numpy.ndarray, dic: dict) -> tuple:
    # Select slices to reconstruct
    try:
        start_slice = int(dic['slices'][0])
        end_slice = int(dic['slices'][1])

        if end_slice < 0:
            logger.info(f'Reconstructing all {tomogram.shape[1]} slices.')
        else:
            tomogram = tomogram[:, start_slice:end_slice, :]
            flat = flat[:, start_slice:end_slice, :]
            dark = dark[:, start_slice:end_slice, :]
            logger.info(f'Reconstructing slices {start_slice} to {end_slice}.')
    except (KeyError, IndexError, TypeError, ValueError):
        logger.info(f'Reconstructing all {tomogram.shape[1]} slices.')

    return tomogram, flat, dark


