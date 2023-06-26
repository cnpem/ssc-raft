from ...rafttypes import *
import numpy as np

from ...tomogram.flatdark import *
from ...tomogram.rotationaxis import *
from ...rings import *
from ...geometries.gc.fdk import *
from ...geometries.gc.em import *

def preprocessing_mogno(tomo, flat, dark, experiment):
   """Preprocessing data from Mogno beamline (conebeam): correction of projection, correction of rotation axis and correction of rings.

    Args:
        data (ndarray): Cone beam projection tomogram. The axes are [angle, slices, lenght].
        flat (ndarray): Single cone beam ray projection. The axes are [number of flats, slices, lenght].
        dark (ndarray): Single dark projection. The axes are [slices, lenght].
        experiment (dictionary): Dictionary with the experiment info.

    Returns:
        (ndarray): Corrected tomogram (3D). The axes are [slices, angles, lenght].

   Dictionary parameters:
        *``experiment['gpu']`` (ndarray): List of gpus for processing. Defaults to [0].
        *``experiment['rings']`` (bool,int): Tuple flag for application of rings removal algorithm. (apply = True, rings block = 2).
        *``experiment['normalize']`` (bool,bool,int,int): Tuple flag for normalization of projection data. ( normalize = True , use log to normalize = True, total number of frames acquired = data.shape[0], index of initial frame to process = 0).
        *``experiment['shift']`` (bool,int): Tuple (is_autoRot = True, value = 0). Rotation shift automatic corrrection (is_autoRot).
        *``experiment['padding']`` (int): Number of elements for horizontal zero-padding. Defaults to 0.
        *``experiment['detectorType']`` (string): If detector type. If 'pco' discard fist 11 rows of data. Defauts to 'pco'.
        *``experiment['findRotationAxis']`` (int,int,int): For rotation axis function. (nx_search=500, nx_window=500, nsinos=None).

   """

   is_normalize = experiment['normalize'][0]

   if experiment['detectorType'] == 'pco': 
      tomo[:, :11,:] = 1.0 
      
      if len(dark.shape) == 2:
         dark[:11,:]   = 0.0
      if len(dark.shape) == 3:
         dark[:,:11,:] = 0.0
      if len(dark.shape) == 4:
         dark[:,:,:11,:] = 0.0

      if len(flat.shape) == 2:
         flat[:11,:]   = 1.0
      if len(flat.shape) == 3:
         flat[:,:11,:] = 1.0
      if len(flat.shape) == 4:
         flat[:,:11,:] = 1.0

   if is_normalize:
      logger.info('Begin Flat and Dark correction')

      experiment['uselog'] = experiment['normalize'][1]
      experiment['frames info'] = (experiment['normalize'][2],experiment['normalize'][3])

      if(len(flat.shape) == 4):
         flats_ = np.zeros((flat.shape[0], flat.shape[2], flat.shape[3]))
         for j in range(flat.shape[0]):
               for i in range(flat.shape[1]):
                  flats_[j,:,:] += flat[j,i,:,:]
                  # flats_[1,:,:] += flat[1,i,:,:]
         flats_ = flats_/flat.shape[1]
         flat   = flats_

      if(len(dark.shape) == 4):
         darks_ = np.zeros((dark.shape[0], dark.shape[2], dark.shape[3]))
         for j in range(dark.shape[0]):
               for i in range(dark.shape[1]):
                  darks_[j,:,:] += dark[j,i,:,:]
                  # darks_[1,:,:] += dark[1,i,:,:]
         darks_ = darks_/dark.shape[1]
         dark = darks_

      tomo = correct_projections(tomo, flat, dark, experiment)
   else:
      logger.info('Not applying Flat and Dark correction')
      tomo = np.swapaxes(tomo,0,1)

   proj = correct_rotation_axis360(np.swapaxes(tomo,0,1), experiment)

   dic                 = {}
   dic['gpu']          = experiment['gpu']
   dic['lambda rings'] = -1 
   dic['rings block']  = experiment['rings'][1]

   if experiment['rings'][0]:
      logger.info('Applying Rings regularization...')
      proj = rings(proj, dic)
      logger.info('Finished Rings regularization...')

   return proj

def reconstruction_mogno(data: np.ndarray, flat: np.ndarray, dark: np.ndarray, experiment: dict) -> np.ndarray:

   """Computes the Reconstruction of a Conical Sinogram for Mogno Beamline.

   Args:
      data (ndarray): Cone beam projection tomogram. The axes are [angle, slices, lenght].
      flat (ndarray): Single cone beam ray projection. The axes are [number of flats, slices, lenght].
      dark (ndarray): Single dark projection. The axes are [slices, lenght].
      experiment (dictionary): Dictionary with the experiment info.

   Returns:
      (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [x, y, z].


   Dictionary parameters:
      *``experiment['z1[m]']`` (float): Source-sample distance in meters. Defaults to 500e-3.
      *``experiment['z1+z2[m]']`` (float): Source-detector distance in meters. Defaults to 1.0.
      *``experiment['detectorPixel[m]']`` (float): Detector pixel size in meters. Defaults to 1.44e-6.
      *``experiment['reconSize']`` (int): Reconstruction dimension. Defaults to data shape[0].
      *``experiment['gpu']`` (ndarray): List of gpus for processing. Defaults to [0].
      *``experiment['method']`` (string): Method for reconstruction: 'em' or 'fdk'. Defaults to 'fdk'.

   Options:
      *``experiment['fourier']`` (bool): Type of filter for reconstruction. True = Fourier, False = Convolution.
      *``experiment['rings']`` (bool,int): Tuple flag for application of rings removal algorithm. (apply = True, rings block = 2).
      *``experiment['normalize']`` (bool,bool,int,int): Tuple flag for normalization of projection data. ( normalize = True , use log to normalize = True, total number of frames acquired = data.shape[0], index of initial frame to process = 0).
      *``experiment['shift']`` (bool,int): Tuple (is_autoRot = True, value = 0). Rotation shift automatic corrrection (is_autoRot).
      *``experiment['padding']`` (int): Number of elements for horizontal zero-padding. Defaults to 0.
      *``experiment['detectorType']`` (string): If detector type. If 'pco' discard fist 11 rows of data. Defauts to 'pco'.
      *``experiment['findRotationAxis']`` (int,int,int): For rotation axis function. (nx_search=500, nx_window=500, nsinos=None).
         
    """

   #Set dictionary parameters by default if not already set.
   dicparams = ('fourier','rings','normalize', 'shift','padding','detectorType','findRotationAxis','method')
   default    = (False, (False, 0), (False, False, 0, 0), (False, 0), 0, 'x', (0, 0, 0),'fdk')
   SetDictionary(experiment,dicparams,default)

   logger.info(f'FDK: Begin preprocessing')

   data = preprocessing_mogno(data, flat, dark, experiment)

   logger.info(f'FDK: Finished preprocessing')

   logger.info(f'FDK: Begin reconstruction')

   recon = fdk(data, experiment)

   logger.info(f'FDK: Finished reconstruction')

   return recon