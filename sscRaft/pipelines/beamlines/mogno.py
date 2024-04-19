from ...rafttypes import *

from ...processing.io import *
from ...filters.phase_filters import *
from ...processing.alignment.rotationaxis import *
from ...processing.rings import *
from ...geometries.conebeam.fdk import *
from ...geometries.parallel.fbp import *
from ...processing.background_correction import *

from sscRaft import __version__

required = ()
default_dic_key = ( 
                    "gpu",
                    "hdf5path",
                    "id sample",
                    "TempPath",
                    "Flathdf5path",
                    "Darkhdf5path",
                    "padding",
                    "norm", 
                    "uselog",
                    "save norm", 
                    "rings", 
                    "lambda rings",
                    "rings block",
                    "save rings",
                    "rotation axis",
                    "shift",
                    "findRotationAxis",
                    "save rot axis",
                    "recon",
                    "method",
                    "slices",
                    "filter", 
                    "end_angle[degrees]",
                    "paganin regularization",
                    "save recon"
                )

default_dic_values = (
                        [0],
                        "scan/detector/data",
                        "",
                        "",
                        "scan/detector/flats",
                        "scan/detector/darks",
                        2,
                        True,
                        True, 
                        False,
                        True, 
                        -1,
                        2,
                        False,
                        False,
                        [True, 0],
                        [400, 400, None],
                        False,
                        True,
                        "fdk",
                        [0,-1],
                        "hamming", 
                        360,
                        0.0,
                        True
                    )

def FlatDarkCorrection_(tomogram, flat, dark, dic):
    """Legacy function. Wrapper for ``FlatDarkCorrection()``
    """
    tomogram = FlatDarkCorrection(tomogram, flat, dark, dic)
    return tomogram

def FlatDarkCorrection(tomogram, flat, dark, dic):
    """Mogno Pipeline function. Applies Flat/Dark (or background) correction.

    Args:
        tomogram (ndarray): Measurement data with shape [angles,slices,lenght]
        flat (ndarray): Flat data with shape [number_of_flats,slices,lenght] or [slices,lenght]
        dark (ndarray): Dark data with shape [1,slices,lenght] or [slices,lenght]
        dic (dict): Dictionary with parameters

    Returns:
        (ndarray): Background corrected tomographic data with shape [slices,angles,lenght]
    """
    start = time()

    try:
        id_sample = dic['id sample']
    except:
        id_sample = ""

    savepath = dic['TempPath'] + 'RaftNorm_' + id_sample + '_' + dic['Iname']

    # Enters [angles,slices,rays]
    tomogram = correct_projections(tomogram, flat, dark, dic)
    # Returns [slices,angles,rays]

    elapsed = time() - start
    logger.info(f'Time for Raft Flat/Dark Correction: {elapsed} seconds')
    start = time()
    logger.info("Finished Raft Flat/Dark Correction")

    if dic['save norm']:
        logger.info("Saving File...")
        save(savepath,tomogram)
        logger.info("Finished Saving")

    return tomogram

def PhaseFilter(tomogram, dic):
   start = time()

   # Enters [angles,slices,rays] 
   outpath = dic['TempPath']
   name = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""
   savepath = outpath + 'RaftPhaseFilter_' + id_sample + '_' + name

   tomogram = phase_filters(tomogram,dic)

   # Returns [angles,slices,rays]

   elapsed = time() - start
   logger.info(f'Time for Raft Phase Filter: {elapsed} seconds')
   start = time()
   logger.info("Finished Raft Phase Filter")

   if dic['save phase filter']:
      logger.info("Saving File...")
      save(savepath,tomogram)
      logger.info("Finished Saving")

   return tomogram


def Rings_function(tomogram, dic):
   start = time()

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

   elapsed = time() - start
   logger.info(f'Time for Raft Rings: {elapsed} seconds')
   start = time()
   logger.info("Finished Raft Rings")

   if dic['save rings']:
      logger.info("Saving File...")
      save(savepath,tomogram)
      logger.info("Finished Saving")

   return tomogram

def rotationAxis(tomogram, dic):
   start = time()

   # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
   outpath  = dic['TempPath']
   name     = dic['Iname']
   try:
      id_sample = dic['id sample']
   except:
      id_sample = ""

   savepath = outpath + 'RaftRotAxis_' + id_sample + '_' + name

   tomogram, _ = correct_rotation_axis360(tomogram, dic)

   # All functions on sscRaft returns [slices,angles,rays] 

   elapsed = time() - start
   logger.info(f'Time for Raft Rotation Axis: {elapsed} seconds')
   start = time()
   logger.info("Finished Raft Rotation Axis")

   if dic['save rot axis']:
      logger.info("Saving File...")
      save(savepath,tomogram)
      logger.info("Finished Saving")
   
   return tomogram

def recon_(tomogram,dic):
    start = time()

    method = dic['method']

    # All functions on sscRaft enters [slices,angles,rays] (EXCEPT correct_projections)
    outpath = dic['TempPath']
    name = dic['Iname']
    try:
        id_sample = dic['id sample']
    except:
        id_sample = ""
        
    savepath = outpath + 'RaftRecon_' + id_sample + '_' + name   

    dic['angles'] = numpy.linspace(0.0, dic['end_angle[degrees]'] * numpy.pi / 180, tomogram.shape[1], endpoint=False)

    if dic['method'] == 'fdk':
        recon = fdk(tomogram, dic)

    elif dic['method'] == 'fbp':
        recon = fbp(tomogram, dic)
        recon = numpy.flip(recon,axis=1)
    else:
        met = dic['method']
        logger.warning(f'Reconstruction method selected ({met}) does not exist on sscRaft package.')
        logger.warning(f'No reconstruction done! Returning Tomogram...')
        recon = tomogram

    # All functions on sscRaft returns [slices,angles,rays] 
    elapsed = time() - start
    logger.info(f'Time for Raft {method} method: {elapsed} seconds')
    logger.info("Finished Raft Reconstruction")

    if dic['save recon']:
        logger.info("Saving File...")
        save(savepath, recon, dic)
        logger.info("Finished Saving")

    return recon

def reconstruction_mogno(param = None):
    r"""Computes Tomographic Reconstruction (Conical and Parallel) considering parameters (param) from a json file

    Args:
        param(sys.argv or list of strings): Arguments from function call ``param = sys.argv`` or ``param = ['-j','file.json','-n','json_key']``

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z,y,x].

        
    Dictionary parameters on json file and key:

        * "Ipath" (str): Absolute path of the data [default: None] [required] 
        * "Iname" (str): Name of the raw data hdf5 file [angles, slices, rays] [default: None] [required] 
        * "hdf5path" (str,optional): Data path inside the raw hdf5 file [default: "scan/detector/data"] 
        * "id sample" (str,optional): Prefix identification of the reconstructed volume [default: ""]
        * "TempPath" (str): Absolute path of the reconstruction to be saved [default: None] [required]
        * "Flatpath" (str,optional): Name and path of the flat hdf5 file [number of flats, slices, rays] [default: "Ipath"+"Iname"] 
        * "Flathdf5path" (str): Flat path inside the raw hdf5 file [default: "scan/detector/flats"] 
        * "Darkpath" (str): Name and path of the dark hdf5 file [number of darks, slices, rays] [default: "Ipath"+"Iname"]  
        * "Darkhdf5path" (str): Dark path inside the raw hdf5 file [default: "scan/detector/darks"]  
        * "gpu" (int list): List of number of gpus for reconstruction. Example for 2 GPUs: [0,1] [default: None] [required]
        * "z1[m]" (float): z1 beamline size in meters [default: None] [required]  
        * "z2[m]" (float,optional): z2 beamline size in meters [default: 0.0]
        * "z1+z2[m]" (float): z1+z2 beamline size in meters [default: None] [required]
        * "detectorPixel[m]" (float): Detector pixel size in meters [default: None] [required]
        * "energy[eV]" (float): Beam line energy in KeV [default: None] [required]
        * "detector" (str): Type of the detector: pco or pimega [default: None] [required]
        * "padding" (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 2] 
        * "norm" (bool,optional): Flat-dark normalization [default: True] 
        * "uselog" (bool,optional): Apply -log on corrected data [default: True]
        * "save norm" (bool,optional): Save the flat-dark norm [default: False] 
        * "rings" (bool,optional): Apply rings [default: True] 
        * "lambda rings" (float,optional): Lambda rings value - Small values between 0 and 1 [default: -1, automatic] 
        * "rings block" (int,optional): Rings block value - Recommended 1,2,4,8 or 16 [default: 2] 
        * "save rings" (bool,optional): Save rings correction [default: False] 
        * "rotation axis" (bool,optional): Apply correction rotaxis [default: True] 
        * "shift" (list: [(bool),(int)], optional): [Auto rotation correction [default: True], Rotation shift value [default: 0]]. Example: [True, 34]
        * "findRotationAxis" (list: [(int),(int),(int)], optional): [Value of width of the search [default: 500], Value of sinogram size will be used in the horizontal axis [default: 500], Number of sinograms to average over [default: None = nslices/2]]
        * "save rot axis" (bool,optional): Save correction rotaxis [default: False] 
        * "recon" (bool,optional): Apply reconstruction [default: True]
        * "method" (str,optional): Choose the reconstruction method: "fdk" or "fbp" [default: fdk]
        * "slices" (list: [(int),(int)],optional): (FBP) Choose the reconstructed slices [start slice, end slice] [default: [0,-1], all slices]
        * "filter" (str,optional): Choose reconstruction filter. Options: "ramp", "gaussian", "hamming", "hann", "cosine", "lorentz", "rectangle", "none" [default: "hamming"]
        * "end_angle[degrees]" (int,optional): Tomography final angle in degrees [default: 360]
        * "paganin regularization" (float,optional): Paganin regularization value ( value >= 0 ) [default: 0.0, no paganin applied]
        * "save recon" (bool,optional): Save reconstruction value [default: True]

    Example of call inside python3: 

    >>> param = ['-j', 'raft.json', '-n', 'recon_example']
    >>> recon = reconstruction_mogno(param)
    
    Example of a python3 script ``raft_script_list.py`` with param = list:
    
    >>> import sscRaft
    >>> param = ['-j', 'raft.json', '-n', 'recon_example']
    >>> recon = sscRaft.reconstruction_mogno(param)

    Example of call from a ``SBATCH`` script on the TEPUI cluster with ``param = sys.argv`` called ``job.srm``:
    
    >>> sbatch job.srm

    Example of ``SBATCH`` script called ``job.srm``:
    
    >>> #!/bin/bash
    >>> #SBATCH --gres=gpu:1      # Total number of GPUs
    >>> #SBATCH -p petro          # Select partition (or fila - queue) to be used
    >>> #SBATCH -J Raft           # Select slurm job name
    >>> #SBATCH -o ./output.out   # Select file path of slurm outputs
    >>> module load python3/3.9.2
    >>> module load cuda/11.2
    >>> module load hdf5/1.12.2_parallel
    >>> python3 -u raft_script.py -j raft.json -n recon_example > output_recon.log 2> error_recon.log
    >>> # sintax (comment):
    >>> #
    >>> # python3             - call python to run
    >>> # -u                  - this arguments prints the log messages while the program is running
    >>> # raft_script.py      - name of the python script
    >>> # -j                  - argument that preceeds the name of the json file (raft.json)
    >>> # raft.json           - name of hte json file
    >>> # -n                  - argument that preceeds the name of the json block inside the json file (raft.json)
    >>> # recon_example       - name of the json block the json file (raft.json) with the reconstruction parameters

    Example of a python3 script ``raft_script.py`` with ``param = sys.argv``:
    
    >>> import sscRaft
    >>> import sys
    >>> param = sys.argv
    >>> recon = sscRaft.reconstruction_mogno(param)

    """
    
    if param == None:
        message_error = f'Missing argument of \'Read_Json()\'.'
        logger.error(message_error)
        logger.error(f'param = [\'-j\',\'name_of_json_file.json\',\'-n\',\'name_of_key_inside_json_file\'].')
        raise ValueError(message_error)
   
    # Set json dictionary:
    dic = Read_Json(param)
    recon = reconstruction_mogno_cli(dic)

    return recon


def reconstruction_mogno_cli(dic:dict):
    """Computes Tomography Reconstruction (Conical and Parallel)

    Args:
        dic (dict): Dictionary with all the parameters for the recon.

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z,y,x].

    Dictionary parameters:
    
        * ``dic['Ipath']`` (str): Absolute path of the data [default: None] [required] 
        * ``dic['Iname']`` (str): Name of the raw data hdf5 file [angles, slices, rays] [default: None] [required] 
        * ``dic['hdf5path']`` (str,optional): Data path inside the raw hdf5 file [default: 'scan/detector/data'] 
        * ``dic['id sample']`` (str,optional): Prefix identification of the reconstructed volume [default: '']
        * ``dic['TempPath']`` (str): Absolute path of the reconstruction to be saved [default: None] [required]
        * ``dic['Flatpath']`` (str,optional): Name and path of the flat hdf5 file [number of flats, slices, rays] [default: 'Ipath'+'Iname'] 
        * ``dic['Flathdf5path']`` (str): Flat path inside the raw hdf5 file [default: 'scan/detector/flats'] 
        * ``dic['Darkpath']`` (str): Name and path of the dark hdf5 file [number of darks, slices, rays] [default: 'Ipath'+'Iname']  
        * ``dic['Darkhdf5path']`` (str): Dark path inside the raw hdf5 file [default: 'scan/detector/darks']  
        * ``dic['z1[m]']`` (float): z1 beamline size in meters [default: None] [required]  
        * ``dic['z2[m]']`` (float,optional): z2 beamline size in meters [default: 0.0]
        * ``dic['z1+z2[m]']`` (float): z1+z2 beamline size in meters [default: None] [required]
        * ``dic['detectorPixel[m]']`` (float): Detector pixel size in meters [default: None] [required]
        * ``dic['energy[eV]']`` (float): Beam line energy in KeV [default: None] [required]
        * ``dic['detector']`` (str): Type of the detector: pco or pimega [default: None] [required]
        * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 2] 
        * ``dic['norm']`` (bool,optional): Flat-dark normalization [default: True] 
        * ``dic['uselog']`` (bool,optional): Apply -log on corrected data [default: True]
        * ``dic['save norm']`` (bool,optional): Save the flat-dark norm [default: False] 
        * ``dic['rings']`` (bool,optional): Apply rings [default: True] 
        * ``dic['lambda rings']`` (float,optional): Lambda rings value - Small values between 0 and 1 [default: -1, automatic] 
        * ``dic['rings block']`` (int,optional): Rings block value - Recommended 1,2,4,8 or 16 [default: 2] 
        * ``dic['save rings']`` (bool,optional): Save rings correction [default: False] 
        * ``dic['rotation axis']`` (bool,optional): Apply correction rotaxis [default: True] 
        * ``dic['shift']`` (list: [(bool),(int)], optional): [Auto rotation correction [default: True], Rotation shift value [default: 0]]. Example: [True, 34]
        * ``dic['findRotationAxis']`` (list: [(int),(int),(int)], optional): [Value of width of the search [default: 500], Value of sinogram size will be used in the horizontal axis [default: 500], Number of sinograms to average over [default: None = nslices/2]]
        * ``dic['save rot axis']`` (bool,optional): Save correction rotaxis [default: False] 
        * ``dic['recon']`` (bool,optional): Apply reconstruction [default: True]
        * ``dic['method']`` (str,optional): Choose the reconstruction method: 'fdk' or 'fbp' [default: 'fdk']
        * ``dic['slices']`` (list: [(int),(int)],optional): (FBP) Choose the reconstructed slices [start slice, end slice] [default: [0,-1], all slices]
        * ``dic['filter']`` (str,optional): Choose reconstruction filter. Options: 'ramp', 'gaussian', 'hamming', 'hann', 'cosine', 'lorentz', 'rectangle', 'none' [default: 'hamming']
        * ``dic['end_angle[degrees]']`` (int,optional): Tomography final angle in degrees [default: 360]
        * ``dic['paganin regularization']`` (float,optional): Paganin regularization value [default: 0.0, no paganin applied]
        * ``dic['save recon']`` (bool,optional): Save reconstruction value [default: True]
    
    """
    start = time()

    dic['z2[m]'] = dic['z1+z2[m]'] - dic['z1[m]']

    tomogram,flat_,dark_ = ReadTomoFlatDark(dic)

    dic = SetDictionary(dic,required,default_dic_key,default_dic_values)

    frame0 = tomogram[0]
    frame1 = tomogram[-1]

    try:
        start_slice = int(dic['slices'][0])
        end_slice   = int(dic['slices'][1])

        if end_slice == -1:
            end_slice = tomogram.shape[1]

        tomogram    = tomogram[:,start_slice:end_slice,:]
        flat_       = flat_[:,start_slice:end_slice,:]
        dark_       = dark_[:,start_slice:end_slice,:]
    except:
        start_slice = 0
        end_slice = tomogram.shape[1]
        logger.info(f'Reconstructing all {tomogram.shape[1]} slices.')
    
    logger.info(f'Reconstructing slices {start_slice} to {end_slice}.')

    if dic['norm']:
        tomogram = FlatDarkCorrection(tomogram, flat_, dark_, dic)
    else:
        logger.info(f'Reading projection data - already with flat/dark correction.')
        path = dic['Ipath']
        name = dic['Iname']

        try:
            hdf5path_data = dic['hdf5path']
        except:
            message_error = f'Missing json entry ""hdf5path"" with the path inside hdf5 file.'
            logger.error(f'Raft Pipeline...')
            logger.error(message_error)
            logger.error(f'Please guarantee that the data has shape (angles,slices,rays).')
            raise ValueError(message_error)

        tomogram = read_data(dic['detector'],path + name, hdf5path_data)
        frame0 = numpy.copy(tomogram[0])
        frame1 = numpy.copy(tomogram[-1])
        flat_  = numpy.ones((tomogram.shape[1],tomogram.shape[-1]))
        dark_  = numpy.zeros((tomogram.shape[1],tomogram.shape[-1]))

        tomogram = numpy.swapaxes(tomogram,0,1)

    if dic['rotation axis'] and dic['shift'][0] == True:
        if dic['end_angle[degrees]'] == 180:
            logger.info(f'Applying automatic function for rotation axis deviation with 180 degrees.')
            dic['shift'][1]  = -Centersino(frame0, frame1, flat_, dark_) # The minus sign here is to adjust the computed value for the rotationAxis() function
        elif dic['end_angle[degrees]'] > 180:
            logger.info(f'Applying automatic function for rotation axis deviation more than 180 degrees.')
            dic['shift'][1]  = DeviationAxis(numpy.swapaxes(tomogram,0,1), dic)
        else:
            dic['shift'][1]  = 0
            logger.warning(f'The automatic function to find the rotation axis deviation does not work for measurements acquired with less than 180 degrees.')
            logger.warning(f'Please, provide the correct value.')
            logger.warning(f'Continuing reconstruction with no deviation value.')

    if dic['rings']:
        tomogram = Rings_function(tomogram, dic)

    if dic['rotation axis']:
        dic['shift'][0] = False
        tomogram = rotationAxis(tomogram, dic)

    if dic['recon']:
        recon = recon_(tomogram,dic)
    else:
        recon = tomogram

    elapsed = time() - start
    logger.info(f'Time for Raft Reconstruction Pipeline: {elapsed} seconds')
    logger.info("Finished Raft Pipeline")

    return recon

def get_reconstruction(tomogram: numpy.ndarray, flat: numpy.ndarray, dark: numpy.ndarray, dic: dict):
    """Computes Tomography Reconstruction (Conical and Parallel)

    Args:
        tomogram (numpy ndarray): Tomogram. The axes are [angles, slices, rays].
        flat (numpy ndarray): Flat. The axes are [number of flats, slices, rays].
        dark (numpy ndarray): Dark. The axes are [number of darks, slices, rays].
        dic (dict): Dictionary with all the parameters for the recon.

    Returns:
        (ndarray): Reconstructed sample 3D object. The axes are [z,y,x].

    Dictionary parameters:
    
       * ``dic['Iname']`` (str): Name of the reconstruction data hdf5 file to be saved [default: None] [required] 
       * ``dic['id sample']`` (str,optional): Prefix identification of the reconstructed volume [default: '']
       * ``dic['TempPath']`` (str): Absolute path of the reconstruction to be saved [default: None] [required]
       * ``dic['gpu']`` (int list): List of number of gpus for reconstruction. Example for 2 GPUs: [0,1] [default: None] [required]
       * ``dic['z1[m]']`` (float): z1 beamline size in meters [default: None] [required]  
       * ``dic['z2[m]']`` (float,optional): z2 beamline size in meters [default: 0.0]
       * ``dic['z1+z2[m]']`` (float): z1+z2 beamline size in meters [default: None] [required]
       * ``dic['detectorPixel[m]']`` (float): Detector pixel size in meters [default: None] [required]
       * ``dic['energy[eV]']`` (float): Beam line energy in KeV [default: None] [required]
       * ``dic['detector']`` (str): Type of the detector: pco or pimega [default: None] [required]
       * ``dic['padding']`` (int,optional): Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 2] 
       * ``dic['norm']`` (bool,optional): Flat-dark normalization [default: True] 
       * ``dic['uselog']`` (bool,optional): Apply -log on corrected data [default: True]
       * ``dic['save norm']`` (bool,optional): Save the flat-dark norm [default: False] 
       * ``dic['rings']`` (bool,optional): Apply rings [default: True] 
       * ``dic['lambda rings']`` (float,optional): Lambda rings value - Small values between 0 and 1 [default: -1, automatic] 
       * ``dic['rings block']`` (int,optional): Rings block value - Recommended 1,2,4,8 or 16 [default: 2] 
       * ``dic['save rings']`` (bool,optional): Save rings correction [default: False] 
       * ``dic['rotation axis']`` (bool,optional): Apply correction rotaxis [default: True] 
       * ``dic['shift']`` (list: [(bool),(int)], optional): [Auto rotation correction [default: True], Rotation shift value [default: 0]]. Example: [True, 34]
       * ``dic['findRotationAxis']`` (list: [(int),(int),(int)], optional): [Value of width of the search [default: 500], Value of sinogram size will be used in the horizontal axis [default: 500], Number of sinograms to average over [default: None = nslices/2]]
       * ``dic['save rot axis']`` (bool,optional): Save correction rotaxis [default: False] 
       * ``dic['recon']`` (bool,optional): Apply reconstruction [default: True]
       * ``dic['method']`` (str,optional): Choose the reconstruction method: 'fdk' or 'fbp' [default: 'fdk']
       * ``dic['slices']`` (list: [(int),(int)],optional): (FBP) Choose the reconstructed slices [start slice, end slice] [default: [0,-1s], all slices]
       * ``dic['filter']`` (str,optional): Choose reconstruction filter. Options: 'ramp', 'gaussian', 'hamming', 'hann', 'cosine', 'lorentz', 'rectangle', 'none' [default: 'hamming']
       * ``dic['end_angle[degrees]']`` (int,optional): Tomography final angle in degrees [default: 360]
       * ``dic['paganin regularization']`` (float,optional): Paganin regularization value, ( value >= 0 ) [default: 0.0, no paganin applied]
       * ``dic['save recon']`` (bool,optional): Save reconstruction value [default: True]

    """
    start = time()

    dic['z2[m]'] = dic['z1+z2[m]'] - dic['z1[m]']
    
    SetDictionary(dic,default_dic_key,default_dic_values)

    frame0 = tomogram[0]
    frame1 = tomogram[-1]

    try:
        start_slice = int(dic['slices'][0])
        end_slice   = int(dic['slices'][1])

        if end_slice == -1:
            end_slice = tomogram.shape[1]
        
        tomogram    = tomogram[:,start_slice:end_slice,:]
        flat        = flat[:,start_slice:end_slice,:]
        dark        = dark[:,start_slice:end_slice,:]
    except:
        start_slice = 0
        end_slice = tomogram.shape[1]
        logger.info(f'Reconstructing all {tomogram.shape[1]} slices.')

    logger.info(f'Reconstructing slices {start_slice} to {end_slice}.')

    if dic['norm']:
        tomogram = FlatDarkCorrection(tomogram, flat, dark, dic)
    else:
        tomogram = numpy.swapaxes(tomogram, 0, 1)

    if dic['rotation axis'] and dic['shift'][0] == True:
        if dic['end_angle[degrees]'] == 180:
            logger.info(f'Applying automatic function for rotation axis deviation with 180 degrees.')
            dic['shift'][1]  = -Centersino(frame0, frame1, flat, dark) # The minus sign here is to adjust the computed value for the rotationAxis() function
        elif dic['end_angle[degrees]'] > 180:
            logger.info(f'Applying automatic function for rotation axis deviation more than 180 degrees.')
            dic['shift'][1]  = DeviationAxis(numpy.swapaxes(tomogram,0,1), dic)
        else:
            dic['shift'][1]  = 0
            logger.warning(f'The automatic function to find the rotation axis deviation does not work for measurements acquired with less than 180 degrees.')
            logger.warning(f'Please, provide the correct value.')
            logger.warning(f'Continuing reconstruction with no deviation value.')

    if dic['rings']:
        tomogram = Rings_function(tomogram, dic)

    if dic['rotation axis']:
        dic['shift'][0] = False
        tomogram = rotationAxis(tomogram, dic)

    if dic['recon']:
        recon = recon_(tomogram,dic)
    else:
        recon = tomogram

    elapsed = time() - start
    logger.info(f'Time for Raft Reconstruction Pipeline: {elapsed} seconds')
    logger.info("Finished Raft Pipeline")

    return recon


def DeviationAxis(tomogram: numpy.ndarray, dic: dict):
   """Computes the deviation of the Rotation Axis of a Tomogram measured in 360 degrees rotation.

   Args:
      tomogram(numpy.ndarray): tomogram corrected by flat/dark. The axes are [angles, slices, x]
      dic(dictionary): dictionary of parameters

   Dictionary parameters:

      * ``dic['shift']`` (bool,int): Tuple (is_autoRot = True, value = 0). Rotation shift automatic corrrection (is_autoRot).
      * ``dic['findRotationAxis']`` (int,int,int): For rotation axis function. (nx_search=500, nx_window=500, nsinos=None).
   
   Returns:
      (int): Rotation axis deviation 

    """
   start = time()

   nx_search  = dic['findRotationAxis'][0]
   nx_window  = dic['findRotationAxis'][1]
   nsinos     = dic['findRotationAxis'][2]

   deviation  = find_rotation_axis_360(tomogram, nx_search = nx_search, nx_window = nx_window, nsinos = nsinos)

   dic['shift'][1] = deviation

   elapsed = time() - start
   logger.info(f'Time for Raft Rotation Axis Deviation: {elapsed} seconds')
   start = time()
   logger.info("Finished Raft Rotation Axis Deviation")

   return deviation

