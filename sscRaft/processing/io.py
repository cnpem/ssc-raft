from ..rafttypes import *
import numpy
from sscRaft import __version__
import time

class HDF5Saver:
    # HDF5Saver('test_dictionary.hdf5').save_data(random_data)
    def __init__(self, filename):
        self.filename = filename

    def save_data(self, data):
        with h5py.File(self.filename, 'w') as f:
            self._save_dict_to_hdf5(f, data)

    def _save_dict_to_hdf5(self, hdf5_group: str, data, parent_key=""):
        for key, value in data.items():
            if isinstance(value, dict):  # Handle nested dictionaries
                self._save_dict_to_hdf5(hdf5_group.create_group(str(parent_key) + "/" + str(key)),
                                        value,
                                        parent_key + "/" + key)
            else:
                hdf5_group.create_dataset(str(parent_key) + "/" + str(key), data=value)



def read_hdf5(filepath, hdf5path, d_type = numpy.float32):
    """Read HDF5 file with h5py.

    Args:
        filepath (str): Full path and name of HDF5 file
        hdf5path (str): Full data path inside HDF5 file
        d_type (numpy dtype, optional): datatype. [Default: numpy.float32]

    Returns:
        (ndarray): numpy array data with d_type as datatype 
    """
    return h5py.File(filepath, "r")[hdf5path][:].astype(d_type)

def save_hdf5_complex(filepath, data, dic = None, software = 'sscRaft', version = __version__):
   
    file = h5py.File(filepath, 'w') # Save HDF5 with h5py

    file.create_dataset("Real"     , data = data.real) # Save real data to HDF5 output file
    file.create_dataset("Imaginary", data = data.imag) # Save imaginary to HDF5 output file

    if dic is not None:
        try:
            # Call function to save the metadata from dictionary 'dic'
            Metadata_hdf5(outputFileHDF5 = file, dic = dic, software = software, version = version)
        except:
            logger.warning("Warning! Cannot save metadata in HDF5 output file.")
            pass

    file.close()

def save_hdf5(filepath, data, dic = None, software = 'sscRaft', version = __version__):
    
    file = h5py.File(filepath, 'w') # Save HDF5 with h5py
    
    file.create_dataset("data", data = data) # Save data to HDF5 output file
    
    if dic is not None:
        try:
            # Call function to save the metadata from dictionary 'dic'
            Metadata_hdf5(outputFileHDF5 = file, dic = dic, software = software, version = version)
        except:
            logger.warning("Warning! Cannot save metadata in HDF5 output file.")
            pass

    file.close()

def save(filepath, data, dic = None, software = 'sscRaft', version = __version__):

    is_complex = numpy.issubdtype(data.dtype, numpy.complexfloating)

    if is_complex:
        save_hdf5_complex(filepath, data, dic, software, version)
    else:
        save_hdf5(filepath, data, dic, software, version)

def SetDic(dic, paramname, deff):
    try:
        dic[paramname]
        
        if type(dic[paramname]) == list:
            for i in range(len(deff)):
                try:
                    dic[paramname][i] 
                except:
                    value = deff[i]
                    logger.info(f'Using default {paramname}:{value}')
                    dic[paramname][i] = value

    except:
        logger.info(f'Using default {paramname}: {deff}.')
        dic[paramname] = deff

def EvalRequiredDic(dic, paramname):
    for key in paramname:
        try:
            dic[key]
        except:
            message_error = f'Missing required dictionary key: {key}'
            logger.info(message_error)
            raise ValueError(message_error)
        
def SetDictionary(dic,required,optional,default):
    
    EvalRequiredDic(dic, required)

    for key in range(len(optional)):
        SetDic(dic,optional[key], default[key])

def Read_Json(param):

   for j in range(len(param)):
      if param[j] == '-n':
         section = param[j+1]
      elif param[j] == '-j':
         name = param[j+1]

   jason = open(name)
   dic = json.load(jason)[section]

   return dic


def Create_Raft_dictionary(param = None):
    """Create sscRaft dictionary from a json file.

    Args:
        param(sys.argv or list of strings): Arguments from function call ``param = sys.argv``; ``param = ['-j','file.json','-n','json_key']``

    Returns:
        (dict): Dictionary with all the sscRaft parameters for reconstruction.

    """
    if param == None:
        logger.error(f'Missing argument of \'Create_Raft_dictionary()\'.')
        logger.error(f'param = [\'-j\',\'name_of_json_file.json\',\'-n\',\'name_of_key_inside_json_file\'].')
        logger.error(f'Finishing run...')
        sys.exit(1)

    for j in range(len(param)):
        if param[j] == '-n':
            section = param[j+1]
        elif param[j] == '-j':
            name = param[j+1]

    jason = open(name)
    dic = json.load(jason)[section]

    return dic

def convert_uint8(data):
   mn = data.min()
   mx = data.max()

   mx -= mn
   I = ((data - mn)/mx) * 255

   return I.astype(numpy.uint8)

def convert_uint16(data):
    mn = data.min()
    mx = data.max()

    mx -= mn
    I = ((data - mn)/mx) * 65535

    return I.astype(numpy.uint16)

def save_raw_uint16(data,filepath):
    # uint16:
    trans  = convert_uint16(data)
    prefix = str(trans.shape[-1]) + "x" + str(trans.shape[-2]) + "x" + str(trans.shape[-3]) + "_" + str(trans.dtype)
    trans.tofile(filepath + '_' + prefix + '.raw')

def save_raw_uint8(data,filepath):
    # uint8:
    trans  = convert_uint8(data)
    prefix = str(trans.shape[-1]) + "x" + str(trans.shape[-2]) + "x" + str(trans.shape[-3]) + "_" + str(trans.dtype)
    trans.tofile(filepath + '_' + prefix + '.raw')

def Metadata_hdf5(outputFileHDF5, dic, software, version):
    """ Function to save metadata from a dictionary, and the name of the softare used and its version
    on a HDF5 file. The parameters names will be save the same as the names from the dictionary.

    Args:
        outputFileHDF5 (h5py.File type or str): The h5py created file or the path to the HDF5 file
        dic (dictionary): A python dictionary containing all parameters (metadata) from the experiment
        software (str): Name of the python module 'software' used
        version (str): Version of python module used (can be called by: ``software.__version__``; example: ``sscFresnel.__version__``)

    """
    dic['Software'] = software
    dic['Version']  = version

    
    if isinstance(outputFileHDF5, str):

        if os.path.exists(outputFileHDF5):
            hdf5 = h5py.File(outputFileHDF5, 'a')
        else:
            hdf5 = h5py.File(outputFileHDF5, 'w')
    else:
        hdf5 = outputFileHDF5

    for key, value in dic.items():

        h5_path = 'Recon Parameters' #os.path.join('Recon Parameters', key)
        hdf5.require_group(h5_path)
        
        if isinstance(value, numpy.ndarray):
            value = numpy.asarray(value)
            try:
                hdf5[h5_path].create_dataset(key, data=value, shape=value.shape)
            except:
                hdf5[h5_path][key] = value
        
        elif isinstance(value, list) or isinstance(value, tuple):
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
            data = numpy.expand_dims(data,0)

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
                flat = numpy.expand_dims(flat,0)

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
                dark = numpy.expand_dims(dark,0)
        
        # dark = dark - np.mean(dark, axis=0, dtype=np.float32)
        return dark #np.swapaxes(dark,0,1)

def ReadTomoFlatDark(dic):
    start = time.time()

    path = dic['Ipath']
    name = dic['Iname']
    
    try:
        hdf5path_data = dic['hdf5path']
    except:
        hdf5path_data = "scan/detector/data"
        logger.info(f'Using default path inside hdf5 for the data as {hdf5path_data}.')

    try: 
        flatpath = dic['Flatpath'] 
    except: 
        flatpath = path+name
    
    try: 
        darkpath = dic['Darkpath'] 
    except: 
        darkpath = path+name    

    try:    
        hdf5path_flat = dic['Flathdf5path']
    except:
        hdf5path_flat = "scan/detector/flats"
    
    try:    
        hdf5path_dark = dic['Darkhdf5path']
    except:
        hdf5path_dark = "scan/detector/darks"    
    
    # Input path and name
    filepath = path + name

    # Read Raw data
    try:
        tomogram = read_data(dic['detector'],filepath, hdf5path_data)
    except:
        logger.error(f'Error reading data in file {filepath}.')
        logger.error(f'Finishing run...')
        raise ValueError(f'Error reading data in file {filepath}.')

    flat = numpy.ones((1,tomogram.shape[1],tomogram.shape[2]))
    
    if flatpath == "":
        logger.warning(f'Reconstruction continues with no Flat-field.')
    else: 
        try:
            flat = read_flat(dic['detector'],flatpath,hdf5path_flat)
        except:
            logger.warning(f'No Flat-field was found in file {flatpath}. Reconstruction continues with no Flat-field.')

    dark = numpy.zeros((1,tomogram.shape[1],tomogram.shape[2]))
    
    if darkpath == "":
        logger.warning(f'Reconstruction continues with no Dark-field.')
         
        try:
            dark = read_dark(dic['detector'],darkpath,hdf5path_dark)
        except:
            logger.warning(f'No Dark-field was found in file {darkpath}. Reconstruction continues with no Dark-field.')
    
    flat_  = flat[0]
    dark_  = dark[0]

    elapsed = time.time() - start
    logger.info(f'Time for Raft Read data, flat and dark: {elapsed} seconds')
    start = time.time()
    logger.info("Finished Raft Read data, flat and dark")

    return tomogram, flat_, dark_