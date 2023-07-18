from ...rafttypes import *
import numpy as np

from ...tomogram.flatdark import *
from ...tomogram.rotationaxis import *
from ...rings import *
from ...geometries.gc.fdk import *
from ...geometries.gc.emc import *

def preprocessing_mogno(data, flat, dark, experiment):
   """Preprocessing data from Mogno beamline (conebeam): correction of projection, correction of rotation axis and correction of rings.

   Args:
      data (ndarray): Cone beam projection tomogram. The axes are [angles, slices, lenght].
      flat (ndarray): Single cone beam ray projection. The axes are [number of flats, slices, lenght].
      dark (ndarray): Single dark projection. The axes are [number of darks, slices, lenght].
      experiment (dictionary): Dictionary with the experiment info.

   Returns:
      (ndarray, int): Corrected tomogram (3D) with axes [slices, angles, lenght]
      and Number of pixels representing the deviation of the center of rotation. 

   Dictionary parameters:
      *``experiment['gpu']`` (ndarray): List of gpus for processing. Defaults to [0].
      *``experiment['rings']`` (bool,float,int): Tuple flag for application of rings removal algorithm. (apply = True, rings regularization = -1 (automatic), rings block = 1).
      *``experiment['normalize']`` (bool,bool,bool): Tuple flag for normalization of projection data. ( normalize = True , use log to normalize = True, remove negative values = False).
      *``experiment['shift']`` (bool,int): Tuple (is_autoRot = True, value = 0). Rotation shift automatic corrrection (is_autoRot).
      *``experiment['padding']`` (int): Number of elements for horizontal zero-padding. Defaults to 0.
      *``experiment['detectorType']`` (string): If detector type. If 'pco' discard fist 11 rows of data. Defauts to 'pco'.
      *``experiment['findRotationAxis']`` (int,int,int): For rotation axis function. (nx_search=500, nx_window=500, nsinos=None).
      *``experiment['method']`` (string): Method for reconstruction: 'em' or 'fdk'. Defaults to 'fdk'.

   """
   tomo = data.copy()
   flat = flat.copy()
   dark = dark.copy()

   is_normalize = experiment['normalize'][0]
   method       = experiment['method']

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

      if(len(flat.shape) == 4):
         flats_ = np.zeros((flat.shape[0], flat.shape[2], flat.shape[3]))
         for j in range(flat.shape[0]):
               for i in range(flat.shape[1]):
                  flats_[j,:,:] += flat[j,i,:,:]
         flats_ = flats_/flat.shape[1]
         flat   = flats_

      if(len(dark.shape) == 4):
         darks_ = np.zeros((dark.shape[0], dark.shape[2], dark.shape[3]))
         for j in range(dark.shape[0]):
               for i in range(dark.shape[1]):
                  darks_[j,:,:] += dark[j,i,:,:]
         darks_ = darks_/dark.shape[1]
         dark = darks_

      tomo = correct_projections(tomo, flat, dark, experiment)

      if experiment['normalize'][2]:
         for i in range(tomo.shape[1]):
            if tomo[:,i,:].min() < 0:
               tomo[:,i,:] = tomo[:,i,:] + np.abs(tomo[:,i,:].min())
   else:
      logger.info('Not applying Flat and Dark correction')
      tomo = np.swapaxes(tomo,0,1)

   if method == 'em':
      shift      = experiment['shift'][1]
      is_autoRot = experiment['shift'][0]
      tomo = np.swapaxes(tomo,0,1)
      if is_autoRot:
         nx_search  = experiment['findRotationAxis'][0]
         nx_window  = experiment['findRotationAxis'][1]
         nsinos     = experiment['findRotationAxis'][2]
         shift = find_rotation_axis_360(tomo, nx_search, nx_window, nsinos)

   elif method == 'fdk':
      tomo_, shift = correct_rotation_axis360(tomo, experiment)
      tomo = np.copy(tomo_)

      dic                 = {}
      dic['gpu']          = experiment['gpu']
      dic['lambda rings'] = experiment['rings'][1] 
      dic['rings block']  = experiment['rings'][2]

      if experiment['rings'][0]:
         logger.info('Applying Rings regularization...')
         tomo = rings(tomo, dic)
         logger.info('Finished Rings regularization...')
   else:
      logger.error(f'Invalid reconstruction method:{method}. Chosse from options `fdk` or `em`.')

   # Garbage Collector
   # lists are cleared whenever a full collection or
   # collection of the highest generation (2) is run
   # collected = gc.collect() # or gc.collect(2)
   # logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')

   return tomo, shift

def reconstruction_mogno(data: np.ndarray, flat: np.ndarray, dark: np.ndarray, experiment: dict) -> np.ndarray:

   """Computes the Reconstruction of a Conical Sinogram for Mogno Beamline.

   Args:
      data (ndarray): Cone beam projection tomogram. The axes are [angle, slices, lenght].
      flat (ndarray): Single cone beam ray projection. The axes are [number of flats, slices, lenght].
      dark (ndarray): Single dark projection. The axes are [number of darks, slices, lenght].
      experiment (dictionary): Dictionary with the experiment info.

   Returns:
      (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [x, y, z].


   Dictionary parameters:
      *``experiment['z1[m]']`` (float): Source-sample distance in meters. Defaults to 500e-3.
      *``experiment['z1+z2[m]']`` (float): Source-detector distance in meters. Defaults to 1.0.
      *``experiment['detectorPixel[m]']`` (float): Detector pixel size in meters. Defaults to 1.44e-6.
      *``experiment['reconSize']`` (int): Reconstruction dimension. Defaults to data shape[0].
      *``experiment['gpu']`` (list): List of gpus for processing. Defaults to [0].
      *``experiment['method']`` (str): Method for reconstruction: 'em' or 'fdk'. Defaults to 'fdk'.

   Options:
      *``experiment['fourier']`` (bool): Define type of filter computation for `FDK` reconstruction. True = FFT, False = Integration (only available for'ramp' filter). Recommend FFT.
      *``experiment['rings']`` (bool,float,int): Tuple flag for application of rings removal algorithm. (apply = True, rings regularization = -1 (automatic), rings block = 1).
      *``experiment['normalize']`` (bool,bool,bool): Tuple flag for normalization of projection data. ( normalize = True , use log to normalize = True, remove negative values = False).
      *``experiment['shift']`` (bool,int): Tuple (is_autoRot = True, value = 0). Rotation axis automatic corrrection (is_autoRot).
      *``experiment['padding']`` (int): Number of elements for horizontal zero-padding. Defaults to 0.
      *``experiment['detectorType']`` (str): If detector type. If 'pco' discard fist 11 rows of data. Defauts to 'pco'.
      *``experiment['findRotationAxis']`` (int,int,int): For rotation axis function. (nx_search=500, nx_window=500, nsinos=None).
      *``experiment['filter']`` (str,optional): Type of filter for reconstruction. 
      Options = ('none','gaussian','lorentz','cosine','rectangle','hann','hamming','ramp'). Default is 'hamming'.
      *``experiment['regularization']`` (float,optional): Value of regularization for reconstruction (FDK or EM/TV), small values. Default is 1.
      *``experiment['iterations']`` (int,int):  Tuple for EM ONLY. Number of iterations and number of integration points for EM/TV.   
    """

   #Set dictionary parameters by default if not already set.
   # dicparams = ('fourier','rings','normalize', 'shift','padding','detectorType','findRotationAxis','method','filter','regularization')
   # default    = (False, (False, -1, 1), (False, False, 0, 0, False), (False, 0), 0, 'x', (0, 0, 0),'fdk','hamming',1)
   # SetDictionary(experiment,dicparams,default)

   method = experiment['method']

   logger.info(f'RECON-MOGNO: Begin preprocessing')

   tomo, shift = preprocessing_mogno(data, flat, dark, experiment)

   if method == 'em' and experiment['shift']:
      experiment['shift'] = shift
      experiment['niterations'] = (experiment['iterations'][0],1,1,experiment['iterations'][1])
      logger.info('RECON-MOGNO: Rotation axis deviation: {}'.format(shift))

   logger.info(f'RECON-MOGNO: Finished preprocessing')

   logger.info(f'RECON-MOGNO: Begin reconstruction')

   if method == 'fdk':
      recon = fdk(tomo, experiment)
   elif method == 'em':
      recon = em_cone(data, flat, dark, experiment)
   else:
      logger.error(f'Invalid reconstruction method:{method}. Choose from options `fdk` or `em`.')


   # Garbage Collector
   # lists are cleared whenever a full collection or
   # collection of the highest generation (2) is run
   # collected = gc.collect() # or gc.collect(2)
   # logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')

   logger.info(f'RECON-MOGNO: Finished reconstruction')

   return recon