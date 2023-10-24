from ...rafttypes import *
import numpy as np

from ...tomogram.flatdark import *
from ...filters.phase_filters import *
from ...tomogram.rotationaxis import *
from ...rings import *
from ...geometries.gc.fdk import *
from ...geometries.gc.emc import *

def read_data(detector,filepath,hdf5path):

        start = time.time()
                
        data = read_hdf5(filepath,hdf5path)

        elapsed = time.time() - start
        print(f'Time for read hdf5 data file: {elapsed} seconds')
        start = time.time()

        dim_data = len(data.shape)

        if detector == 'pco': 
                # For PCO detector only
                if dim_data == 2:
                        data[:11,:]   = 1.0
                if dim_data == 3:
                        data[:,:11,:] = 1.0
        if dim_data == 2:
                data = np.expand_dims(data,0)

        return data #np.swapaxes(data,0,1)

def read_flat(detector,filepath,hdf5path):

        start = time.time()
                
        flat = read_hdf5(filepath,hdf5path)

        elapsed = time.time() - start
        print(f'Time for read hdf5 flat file: {elapsed} seconds')
        start = time.time()

        dim_flat = len(flat.shape)

        if detector == 'pco': 
                # For PCO detector only
                if dim_flat == 2:
                        flat[:11,:]   = 1.0
                if dim_flat == 3:
                        flat[:,:11,:] = 1.0
                if dim_flat == 4:
                        flat[:,:,:11,:] = 1.0
        if dim_flat == 4:
                flat = flat[:,0,:,:]
        if dim_flat == 2:
                flat = np.expand_dims(flat,0)

        # flat = flat - np.mean(flat, axis=0, dtype=np.float32)
        return flat #np.swapaxes(flat,0,1)

def read_dark(detector,filepath,hdf5path):

        start = time.time()
                
        dark = read_hdf5(filepath,hdf5path)

        elapsed = time.time() - start
        print(f'Time for read hdf5 dark file: {elapsed} seconds')
        start = time.time()

        dim_dark = len(dark.shape)

        if detector == 'pco': 
                # For PCO detector only
                if dim_dark == 2:
                        dark[:11,:]   = 0.0
                if dim_dark == 3:
                        dark[:,:11,:] = 0.0
                if dim_dark == 4:
                        dark[:,:,:11,:] = 0.0
        if dim_dark == 4:
                dark = dark[:,0,:,:]
        if dim_dark == 2:
                dark = np.expand_dims(dark,0)
        
        # dark = dark - np.mean(dark, axis=0, dtype=np.float32)
        return dark #np.swapaxes(dark,0,1)

def convert_uint8(data):
   mn = data.min()
   mx = data.max()

   mx -= mn
   I = ((data - mn)/mx) * 255

   return I.astype(np.uint8)

def Read_Json(param):

   for j in range(len(param)):
      if param[j] == '-n':
         section = param[j+1]
      elif param[j] == '-j':
         name = param[j+1]

   jason = open(name)
   dic = json.load(jason)[section]

   return dic

def read_hdf5(filepath,hdf5path):
   data = h5py.File(filepath, "r")[hdf5path][:].astype(np.float32)

   return data

def save_hdf5(filepath, recon):
   file = h5py.File(filepath, 'w') # Save HDF5 with h5py
   file.create_dataset("data", data = recon) # Save reconstruction to HDF5 output file
   file.close()

def save_hdf5_tomo(filepath, recon, dic):
   file = h5py.File(filepath, 'w') # Save HDF5 with h5py
   file.create_dataset("data", data = recon) # Save reconstruction to HDF5 output file
   try:
      # Call function to save the metadata from dictionary 'dic' with the software 'sscRaft' and its version 'sscRaft.__version__'
      Metadata_hdf5(outputFileHDF5 = file, dic = dic, software = 'sscRaft', version = '2.2.2')
   except:
      print("Error! Cannot save metadata in HDF5 output file.")
      pass

   file.close()

# if dic['norm']:
def _FlatDarkCorrection(dic):
   start = time.time()

   path = dic['Ipath']
   name = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""
   
   outpath = dic['TempPath']

   hdf5path_data = "scan/detector/data"
   hdf5path_flat = "scan/detector/flats"
   hdf5path_dark = "scan/detector/darks"
   
   # Input path and name
   filepath = path + name
   savepath = outpath + 'Norm_' + id_sample + '_' + name

   # Read Raw data
   tomogram = read_data(dic['detector'],filepath, hdf5path_data)

   flat = read_flat(dic['detector'],filepath,hdf5path_flat)

   dark = read_dark(dic['detector'],filepath,hdf5path_dark)

   # Enters [angles,slices,rays]
   tomogram = correct_projections(tomogram, flat[0], dark, dic)
   # Returns [slices,angles,rays]

   elapsed = time.time() - start
   print(f'Time for Flat/Dark Correction: {elapsed} seconds')
   start = time.time()
   print("Finished Flat/Dark Correction")

   if dic['save norm']:
      print("Saving File...")
      save_hdf5(savepath,tomogram)
      print("Finished Saving")

   return tomogram

# if dic['phase']:
def _PhaseFilter(tomogram, dic):
   start = time.time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath = dic['TempPath']
   name = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""
   savepath = outpath + 'PhaseFilter_' + id_sample + '_' + name

   tomogram = phase_filters(tomogram,dic)

   # All functions on sscRaft returns [slices,angles,rays] (EXCEPT correct_projections)

   elapsed = time.time() - start
   print(f'Time for Phase Filter: {elapsed} seconds')
   start = time.time()
   print("Finished Phase Filter")

   if dic['save phase filter']:
      print("Saving File...")
      save_hdf5(savepath,tomogram)
      print("Finished Saving")

   return tomogram


# if dic['rings']:
def _Rings(tomogram, dic):
   start = time.time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath = dic['TempPath']
   name = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""

   savepath = outpath + 'Rings_' + id_sample + '_' + name

   tomogram = rings(tomogram, dic)

   # All functions on sscRaft returns [slices,angles,rays] (EXCEPT correct_projections)

   elapsed = time.time() - start
   print(f'Time for Rings: {elapsed} seconds')
   start = time.time()
   print("Finished Rings")

   if dic['save rings']:
      print("Saving File...")
      save_hdf5(savepath,tomogram)
      print("Finished Saving")

   return tomogram

# Rotation Axis Correction
# if dic['rotation axis']:
def _rotationAxis(tomogram, dic):
   start = time.time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath  = dic['TempPath']
   name     = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""

   savepath = outpath + 'RotAxis_' + id_sample + '_' + name

   tomogram, _ = correct_rotation_axis360(tomogram, dic)

   # All functions on sscRaft returns [slices,angles,rays] (EXCEPT correct_projections)

   elapsed = time.time() - start
   print(f'Time for Rotation Axis: {elapsed} seconds')
   start = time.time()
   print("Finished Rotation Axis")

   if dic['save rot axis']:
      print("Saving File...")
      save_hdf5(savepath,tomogram)
      print("Finished Saving")
   
   return tomogram

# if dic['recon']:
def _recon(tomogram,dic):
   start = time.time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath = dic['TempPath']
   name = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""
      
   savepath = outpath + 'Recon_' + id_sample + '_' + name   

   dic['angles'] = np.linspace(0.0, dic['end_angle[degrees]'] * np.pi / 180, tomogram.shape[1], endpoint=False)

   recon = fdk(tomogram, dic)

   # All functions on sscRaft returns [slices,angles,rays] (EXCEPT correct_projections)

   elapsed = time.time() - start
   print(f'Time for FDK: {elapsed} seconds')
   print("Finished Reconstruction")

   if dic['save recon']:
      print("Saving File...")
      save_hdf5_tomo(savepath, recon, dic)
      print("Finished Saving")

   return recon

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
      if len(tomo.shape) == 2: 
         tomo[:11,:] = 1.0 
      if len(tomo.shape) == 3:
         tomo[:,:11,:] = 1.0

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
         flat[:,:,:11,:] = 1.0

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
      del(tomo_)

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

def reconstruction_mogno2(data: np.ndarray, flat: np.ndarray, dark: np.ndarray, experiment: dict) -> np.ndarray:

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
      recon = em_cone(tomo, flat, dark, experiment)
   else:
      logger.error(f'Invalid reconstruction method:{method}. Choose from options `fdk` or `em`.')


   logger.info(f'RECON-MOGNO: Finished reconstruction')

   return recon

def reconstruction_mogno(param = sys.argv):

   """Computes the Reconstruction of a Conical Sinogram for Mogno Beamline.

   Args:
      param(sys.argv): Arguments from function call param = sys.argv

   Returns:
      (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [x, y, z].

    """
   start = time.time()

   # Set json dictionary:
   dic = Read_Json(param)

   dic['z2[m]'] = dic['z1+z2[m]'] - dic['z1[m]']

   if dic['norm']:
      if dic['phase']:
         dic['uselog'] = False
      else:
         dic['uselog'] = True
      tomogram = _FlatDarkCorrection(dic)
   # else:
   #    tomogram = h5py.File(dic['Ipath']+dic['Iname'], "r")["data"][:].astype(np.float32)
   
   if dic['rotation axis'] and dic['shift'][0] == True:

      nx_search  = dic['findRotationAxis'][0]
      nx_window  = dic['findRotationAxis'][1]
      nsinos     = dic['findRotationAxis'][2]
      deviation  = find_rotation_axis_360(np.swapaxes(tomogram,0,1), nx_search = nx_search, nx_window = nx_window, nsinos = nsinos)
      dic['shift'][1] = deviation

   if dic['phase']:
      tomogram = _PhaseFilter(tomogram, dic)

   if dic['rings']:
      tomogram = _Rings(tomogram, dic)

   if dic['rotation axis']:
      dic['shift'][0] = False
      tomogram = _rotationAxis(tomogram, dic)

   if dic['recon']:
      recon = _recon(tomogram,dic)

   elapsed = time.time() - start
   print(f'Time for Reconstruction Pipeline: {elapsed} seconds')
   print("Finished Pipeline")

   return recon

def getDeviationAxis(param = sys.argv):

   """Computes the Reconstruction of a Conical Sinogram for Mogno Beamline.

   Args:
      param(sys.argv): Arguments from function call param = sys.argv

   Returns:
      (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [x, y, z].

    """

   # Set json dictionary:
   dic = Read_Json(param)

   dic['z2[m]'] = dic['z1+z2[m]'] - dic['z1[m]']

   dic['uselog'] = True
   tomogram = _FlatDarkCorrection(dic)

   dic['shift'][0] = True
   dic['shift'][1] = 0

   nx_search  = dic['findRotationAxis'][0]
   nx_window  = dic['findRotationAxis'][1]
   nsinos     = dic['findRotationAxis'][2]
   
   deviation  = find_rotation_axis_360(np.swapaxes(tomogram,0,1), nx_search = nx_search, nx_window = nx_window, nsinos = nsinos)

   return deviation

