from ...rafttypes import *

from ...processing.io import *
from ...phase_retrieval.phase import *
from ...processing.alignments.rotationaxis import *
from ...processing.rings import *

from ..constructors import process_tomogram_volume
from ...processing.background_correction import *
from ..recon_python_pipeline import reconstruction_methods
from ..pre_process import (select_slices_to_reconstruct,angle_verification,fix_rotation_axis,
                          find_stitching_auto,calculate_transposed_volumes,
                          update_paganin_regularization,find_rotation_axis_auto,
                          fix_stitching)

from ..filters_pipeline import multiple_filters
from ...processing.opt import transpose, flip_x, flip_x_np

numpy.seterr(divide='ignore', invalid='ignore') #to ignore divde by zero warning

import time

def process_tomcat_data(is_stitching, filepaths, h5path, pin_memory=False):
    start_read = time.time()

    data_path    = filepaths[0]
    flat_path    = filepaths[1]
    dark_path    = filepaths[2]

    h5data       = h5path[0]
    h5flat_pre   = h5path[1]
    h5flat_post  = h5path[2]
    h5dark       = h5path[3]

    data         = read_hdf5(data_path, h5data, d_type=numpy.float32, pin_memory=pin_memory)
    flat_pre     = read_hdf5(flat_path, h5flat_pre, d_type=numpy.float32)
    flat_post    = read_hdf5(flat_path, h5flat_post, d_type=numpy.float32)
    dark         = read_hdf5(dark_path, h5dark, d_type=numpy.float32)

    alloc_output_start = time.time()

    if is_stitching == 'T':
        recon_shape = (data.shape[1], 2 * data.shape[2], 2 * data.shape[2])
    else:
        recon_shape = (data.shape[1], data.shape[2], data.shape[2])

    if pin_memory:
        recon = pinned_empty(recon_shape, dtype=numpy.float32)
    else:
        recon = numpy.zeros(recon_shape, dtype=numpy.float32)

    elapsed = time.time() - alloc_output_start
    logger.info(f"Alloc recon output: {elapsed:.2f} seconds.")

    elapsed = time.time() - start_read
    logger.info(f"Read datasets (data, flats, dark, angles): {elapsed:.2f} seconds.")

    start_pre = time.time()

    if pin_memory:
        flats = pinned_empty((flat_pre.shape[-2], 2, flat_pre.shape[-1]),
                                    dtype=numpy.float32)
    else:
        flats = numpy.empty((flat_pre.shape[-2], 2, flat_pre.shape[-1]),
                            dtype=numpy.float32)

    flats[:, 0, :] = numpy.mean(flat_pre, axis=0)
    flats[:, 1, :] = numpy.mean(flat_post, axis=0)
    darks          = numpy.mean(dark, axis=0)

    data = transpose(data[:-1, :, :])

    data  = flip_x(data)
    flats = flip_x_np(flats)
    darks = flip_x_np(darks)

    logger.info(f'Data shape (nangles,nslices,nrays): {data.shape}')

    elapsed = time.time() - start_pre
    logger.info(f"Pre-Process TOMCAT data and flat for RAFT: {elapsed:.2f} seconds.")
    
    return data, flats, darks, recon

def tomcat_reconstruction_pipeline(tomogram: numpy.ndarray, flat: numpy.ndarray, dark: numpy.ndarray,
                                   recon: numpy.ndarray, dic: dict) -> numpy.ndarray:

    start = time.time()

    gpus         = dic.get('gpu',[0])
    is_stitching = dic.get('stitching', 'F')

    # Select slices (angles) to reconstruct
    # tomogram, flat, dark = select_slices_to_reconstruct(tomogram=tomogram, flat=flat, dark=dark, dic=dic)

    # Angle verification
    angle_vector = angle_verification(dic=dic, tomogram=tomogram)
    
    dic['angles[rad]'] = angle_vector 

    logger.debug(f"Tomogram shape: {tomogram.shape}, Flat shape: {flat.shape}, Dark shape: {dark.shape}")
    
    deviation = dic.get('axis offset', -1)

    if is_stitching == 'F': 
        if deviation == -1:
            # Find automatic rotation axis deviation:
            deviation = find_rotation_axis_auto(dic=dic, angles=angle_vector, data=tomogram, flat=flat[:,0,:], dark=dark)
            dic['axis offset'], dic['axis offset auto']  = deviation, False

    # Correction by flat-dark:
    start2 = time.time()
    tomogram = correct_background(tomogram, flat, dark, gpus, True)
    elapsed = time.time() - start2
    logger.info(f'Finished correction by Flat, Dark and log! Total Time: {elapsed:.2f} seconds')

    # Stitching
    if is_stitching == 'T':
        offset = find_stitching_auto(dic=dic, tomogram=tomogram)
        dic['stitching overlap'] = offset
        tomogram = process_tomogram_volume(tomogram, recon, dic, fix_stitching)
        dic['angles[rad]'] = numpy.linspace(0.0, numpy.pi, tomogram.shape[1], endpoint=False)

    tomogram = process_tomogram_volume(tomogram, recon, dic, multiple_filters)
    dic      = update_paganin_regularization(dic) 

    # Perform rotation axis deviation correction
    if is_stitching == 'F':
        dic['rotation axis offset'] = dic['axis offset']
    else:
        dic['rotation axis offset'] = 0

    # Perform tomography reconstruction
    recon = process_tomogram_volume(tomogram, recon, dic, reconstruction_methods)

    elapsed = time.time() - start
    logger.info(f'Finished TOMCAT Reconstruction Pipeline! Total Time: {elapsed:.2f} seconds')

    return recon

def tomcat_free_pinned_memory(tomogram: numpy.ndarray, flats: numpy.ndarray, recon: numpy.ndarray):
    free_pinned_array(flats)
    free_pinned_array(tomogram)
    free_pinned_array(recon)

def tomcat_pipeline(dic: dict) -> None:

    pin_memory   = dic.get('pin_memory',False)
    is_stitching = dic.get('stitching','F')

    filepath     = dic.get('input_path','') 
    filename     = dic.get('input_name','')

    datafile     = os.path.join(filepath,filename)

    flat_path    = dic.get('flat_path', datafile)
    dark_path    = dic.get('dark_path', datafile)

    h5data       = dic.get('input_data_hdf5_path','exchange/data') 
    h5flat_pre   = dic.get('flat_pre_dataset','exchange/data_white_pre') 
    h5flat_pos   = dic.get('flat_pos_dataset','exchange/data_white_post') 
    h5dark       = dic.get('dark_dataset','exchange/data_dark') 

    filepaths    = [datafile,flat_path,dark_path]
    h5path       = [h5data,h5flat_pre,h5flat_pos,h5dark]

    tomogram, flat, dark, recon = process_tomcat_data(is_stitching, filepaths, h5path, pin_memory=pin_memory)

    recon = tomcat_reconstruction_pipeline(tomogram, flat, dark, recon, dic)

    if pin_memory:
        tomcat_free_pinned_memory(tomogram, flat, recon)