import numpy as np
import h5py
import sscRaft
from sscIO import io
import time

# Dicionário com parâmetros 
experiment = {}
experiment['z1 [m]'] = 2103*1e-3
experiment['z1+z2 [m]'] = 2530.08*1e-3
experiment['detector pixel [m]'] = 3.61*1e-6
experiment['recon size'] = 2048
experiment['gpus'] = np.array([0,1,2,3])
experiment['rings'] = (True,2)
experiment['normalize'] = (True,True)
experiment['padding'] = 800
experiment['shift'] = (True,0)
experiment['pco detector'] = True


start_total = time.time()

# Caminho dado entrada
in_path = '/ibira/lnls/beamlines/mogno/proposals/20221835/data/dentes_experiment/dente_art_ep/'
in_name = 'tomo_dente_art_ep_2x_eps_3468nm_z1_2440_z1z2_2504_0_05_Si_000.hdf5'

# Caminho dado saida
out_path = ''
out_name = 'recon_mogno_rings_test.h5'

start_read = time.time()

# Ler dado HDF5
data = h5py.File(in_path + in_name, "r")["scan"]["detector"]["data"][:].astype(np.float32)
flat = h5py.File(in_path + in_name, "r")["scan"]["detector"]["flats"][:].astype(np.float32)[0,:,:]
dark = h5py.File(in_path + in_name, "r")["scan"]["detector"]["darks"][:].astype(np.float32)[0,:,:]

elapsed_read = time.time() - start_read

# Reconstrução por FDK
recon = sscRaft.reconstruction_fdk(experiment, data, flat, dark)

# Adicionar parametros no dicionario (experiment) para salvar como metadado
experiment['Input file'] = in_path + in_name
experiment['Energy [KeV]'] = '22 and 39'
experiment['Software'] = 'sscRaft'
experiment['Version'] = sscRaft.__version__

start_write = time.time()

# # Salvar HDF5 com h5py
file = h5py.File(out_path + out_name, 'w')

# Save reconstruction to HDF5 output file
file.create_dataset("data", data = recon)

file.close()

elapsed_write = time.time() - start_write

start_meta = time.time()

# # Salvar metadado no arquivo hdf5
file = h5py.File(out_path + out_name, 'a')

try:
      # Call function to save the metadata from dictionary 'experiment' with the software 'sscRaft' and its version 'sscRaft.__version__'
      sscRaft.Metadata_hdf5(outputFileHDF5 = file, dic = experiment, software = 'sscRaft', version = sscRaft.__version__)
except:
      print("Error! Cannot save metadata in HDF5 output file.")
      pass

file.close()

elapsed_meta = time.time() - start_meta


elapsed_total = time.time() - start_total

print("time read:",elapsed_read)
print("time write:",elapsed_write)
print("time meta:",elapsed_meta)
print("time total:",elapsed_total)

