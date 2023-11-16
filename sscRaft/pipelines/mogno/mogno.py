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
        logger.info(f'Time for read hdf5 data file: {elapsed} seconds')
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
        logger.info(f'Time for read hdf5 flat file: {elapsed} seconds')
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
        logger.info(f'Time for read hdf5 dark file: {elapsed} seconds')
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
      Metadata_hdf5(outputFileHDF5 = file, dic = dic, software = 'sscRaft', version = '2.2.3')
   except:
      logger.error(f"Error! Cannot save metadata in HDF5 output file.")
      pass

   file.close()

def FlatDarkCorrection(dic):
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
   savepath = outpath + 'RaftNorm_' + id_sample + '_' + name

   # Read Raw data
   try:
      tomogram = read_data(dic['detector'],filepath, hdf5path_data)
   except:
      logger.error(f'No detector acquired data was found in file {filepath}.')
      logger.error(f'Finishing run...')
      exit(1)

   try:
      flat = read_flat(dic['detector'],filepath,hdf5path_flat)
   except:
      logger.warning(f'No Flat-field was found in file {filepath}. Reconstruction continues with no Flat-field.')
      flat = np.ones((1,tomogram.shape[1],tomogram.shape[2])) 

   try:
      dark = read_dark(dic['detector'],filepath,hdf5path_dark)
   except:
      logger.warning(f'No Dark-field was found in file {filepath}. Reconstruction continues with no Dark-field.')
      dark = np.zeros((1,tomogram.shape[1],tomogram.shape[2])) 

   # Enters [angles,slices,rays]
   tomogram = correct_projections(tomogram, flat[0], dark, dic)
   # Returns [slices,angles,rays]

   elapsed = time.time() - start
   logger.info(f'Time for Raft Flat/Dark Correction: {elapsed} seconds')
   start = time.time()
   logger.info("Finished Raft Flat/Dark Correction")

   if dic['save norm']:
      logger.info("Saving File...")
      save_hdf5(savepath,tomogram)
      logger.info("Finished Saving")

   return tomogram

def PhaseFilter(tomogram, dic):
   start = time.time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath = dic['TempPath']
   name = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""
   savepath = outpath + 'RaftPhaseFilter_' + id_sample + '_' + name

   tomogram = phase_filters(tomogram,dic)

   # All functions on sscRaft returns [slices,angles,rays] (EXCEPT correct_projections)

   elapsed = time.time() - start
   logger.info(f'Time for Raft Phase Filter: {elapsed} seconds')
   start = time.time()
   logger.info("Finished Raft Phase Filter")

   if dic['save phase filter']:
      logger.info("Saving File...")
      save_hdf5(savepath,tomogram)
      logger.info("Finished Saving")

   return tomogram


def Rings_function(tomogram, dic):
   start = time.time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath = dic['TempPath']
   name = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""

   savepath = outpath + 'RaftRings_' + id_sample + '_' + name

   tomogram = rings(tomogram, dic)

   # All functions on sscRaft returns [slices,angles,rays] (EXCEPT correct_projections)

   elapsed = time.time() - start
   logger.info(f'Time for Raft Rings: {elapsed} seconds')
   start = time.time()
   logger.info("Finished Raft Rings")

   if dic['save rings']:
      logger.info("Saving File...")
      save_hdf5(savepath,tomogram)
      logger.info("Finished Saving")

   return tomogram

def rotationAxis(tomogram, dic):
   start = time.time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath  = dic['TempPath']
   name     = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""

   savepath = outpath + 'RaftRotAxis_' + id_sample + '_' + name

   tomogram, _ = correct_rotation_axis360(tomogram, dic)

   # All functions on sscRaft returns [slices,angles,rays] (EXCEPT correct_projections)

   elapsed = time.time() - start
   logger.info(f'Time for Raft Rotation Axis: {elapsed} seconds')
   start = time.time()
   logger.info("Finished Raft Rotation Axis")

   if dic['save rot axis']:
      logger.info("Saving File...")
      save_hdf5(savepath,tomogram)
      logger.info("Finished Saving")
   
   return tomogram

def recon_(tomogram,dic):
   start = time.time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath = dic['TempPath']
   name = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""
      
   savepath = outpath + 'RaftRecon_' + id_sample + '_' + name   

   dic['angles'] = np.linspace(0.0, dic['end_angle[degrees]'] * np.pi / 180, tomogram.shape[1], endpoint=False)

   recon = fdk(tomogram, dic)

   # All functions on sscRaft returns [slices,angles,rays] (EXCEPT correct_projections)

   elapsed = time.time() - start
   logger.info(f'Time for Raft FDK: {elapsed} seconds')
   logger.info("Finished Raft Reconstruction")

   if dic['save recon']:
      logger.info("Saving File...")
      save_hdf5_tomo(savepath, recon, dic)
      logger.info("Finished Saving")

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

   try:
      phase = dic['phase filter']
   except:
      phase = False

   if dic['norm']:
      if phase:
         dic['uselog'] = False
      else:
         dic['uselog'] = True
      tomogram = FlatDarkCorrection(dic)

   else:
      path = dic['Ipath']
      name = dic['Iname']

      try:
         hdf5path_data = dic['hdf5path']
      except:
         logger.errorprint(f'Raft Pipeline...')
         logger.error(f'Missing json entry ""hdf5path"" with the path inside hdf5 file.')
         logger.error(f'Please guarantee that the data has entries: (slices,angles,rays).')
         logger.error(f'Finishing run...')
         sys.exit(1)

      tomogram = read_data(dic['detector'],path + name, hdf5path_data)

   if dic['rotation axis'] and dic['shift'][0] == True:
         dic['shift'][1]  = DeviationAxis(np.swapaxes(tomogram,0,1), dic)

   if phase:
      tomogram = PhaseFilter(tomogram, dic)

   if dic['rings']:
      tomogram = Rings_function(tomogram, dic)

   if dic['rotation axis']:
      dic['shift'][0] = False
      tomogram = rotationAxis(tomogram, dic)

   if dic['recon']:
      recon = recon_(tomogram,dic)
   else:
      recon = tomogram

   elapsed = time.time() - start
   logger.info(f'Time for Raft Reconstruction Pipeline: {elapsed} seconds')
   logger.info("Finished Raft Pipeline")

   return recon

def getReconstructionProcessing(tomogram: np.ndarray, axis_deviation: int, dic: dict):

   """Computes the Reconstruction of a Conical Sinogram for Mogno Beamline.

   Args:
      tomogram (numpy ndarray): Tomogram. The axes are [slices, angles, rays].

   Returns:
      (ndarray): Reconstructed sample object with dimension n^3 (3D). The axes are [x, y, z].

    """
   start = time.time()

   dic['shift'][1] = axis_deviation
   
   if dic['rings']:
      tomogram = Rings_function(tomogram, dic)

   if dic['rotation axis']:
      dic['shift'][0] = False
      tomogram = rotationAxis(tomogram, dic)

   if dic['recon']:
      recon = recon_(tomogram,dic)
   else:
      recon = tomogram

   elapsed = time.time() - start
   logger.info(f'Time for Raft Reconstruction Pipeline: {elapsed} seconds')
   logger.info("Finished Raft Pipeline")

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
   tomogram = FlatDarkCorrection(dic)

   dic['shift'][0] = True
   dic['shift'][1] = 0
   
   deviation  = DeviationAxis(np.swapaxes(tomogram,0,1), dic)

   return deviation

def DeviationAxis(tomogram: np.ndarray, dic: dict):

   """Computes the deviation of the Rotation Axis of a Tomogram measured in 360 degrees rotation.

   Args:
      tomogram(numpy.ndarray): tomogram corrected by flat/dark. The axes are [angles, slices, x]
      dic(dictionary): dictionary of parameters

   Dictionary parameters:
      *``dic['shift']`` (bool,int): Tuple (is_autoRot = True, value = 0). Rotation shift automatic corrrection (is_autoRot).
      *``dic['findRotationAxis']`` (int,int,int): For rotation axis function. (nx_search=500, nx_window=500, nsinos=None).
   
   Returns:
      (int): Rotation axis deviation 

    """
   start = time.time()

   nx_search  = dic['findRotationAxis'][0]
   nx_window  = dic['findRotationAxis'][1]
   nsinos     = dic['findRotationAxis'][2]

   deviation  = find_rotation_axis_360(tomogram, nx_search = nx_search, nx_window = nx_window, nsinos = nsinos)

   dic['shift'][1] = deviation

   elapsed = time.time() - start
   logger.info(f'Time for Raft Rotation Axis Deviation: {elapsed} seconds')
   start = time.time()
   logger.info("Finished Raft Rotation Axis Deviation")

   return deviation

