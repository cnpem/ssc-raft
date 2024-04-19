---
title: Mogno Reconstructions
description: Reconstruction algorithms for Mogno beamline with cone-beam geometry
sidebar_position: 3
---

:::note
For documentation on functions see [sscRaft documentation](https://gcc.lnls.br/ssc/ssc-raft/index.html).
:::

## Mogno Pipeline

The `sscRaft` includes functions specifically to process data from the Mogno beamline at LNLS. It includes usual pre-processing, reconstruction and post-processing steps onto a unique `Python` wrapper function. It does not includes, as of now, `I.O.` functions. It receives data on memory as a `numpy ndarray` and returns the reconstructed data as a `numpy ndarray`. 

This pipeline method consists of:

- Dictionary input with all parameters
- Raw data _flat_ and _dark_ corrections 
- Correction of rotation axes deviations
- Application of rings filter
- Reconstruction with methods `FDK` and `EM` for cone-beam geometry

### Function call (version 2.2.0)

```python
import sscRaft
'''set dictionary and arrays:

dic = {}
...

data = ...
flat = ...
dark = ...
'''
recon = sscRaft.reconstruction_mogno(data, flat, dark, dic)
```

### Dictionary parameters (version 2.2.0)

- `dic['gpu'] (ndarray)`: List of gpus for processing. 
- `dic['z1[m]'] (float)`: Source-sample distance in meters. 
- `dic['z1+z2[m]'] (float)`: Source-detector distance in meters. 
- `dic['detectorPixel[m]'] (float)`: Detector pixel size in meters. 
- `dic['reconSize'] (int)`: Reconstruction dimension.
- `dic['method'] (str)`: Method for reconstruction.
       - Options are `'em'` or `'fdk'`.

- `dic['fourier'] (bool)`: Define type of filter computation for `FDK` reconstruction. True = FFT, False = Integration. Recommend FFT.
      - `False`: Integration computation of filter is available only for `'ramp'` filter type.
- `dic['normalize'] (bool,bool,bool)`: Tuple flag for normalization of projection data.
    1. First entry: To apply (`True`) or not apply (`False`) _flat_ and _dark_ correction.
    2. Second entry: To apply (`True`) or not apply (`False`) to return the _logarithm_ of the correction.
    3. Third entry: To apply (`True`) or not apply (`False`) the removal of negative values from the corrected data.
- `dic['shift'] (bool,int)`: Tuple. Rotation axis automatic corrrection.
    1. First entry: To apply (`True`) or not apply (`False`) **automatic** rotation axis deviation correction.
    2. Second entry: Value of rotation axis deviation if it is already know.  
- `dic['findRotationAxis'] (int,int,int)`: Extra parameters for rotation axis function.
    1. First entry:  Value of width of the search.
    2. Second entry: Value of sinogram size will be used in the horizontal axis.
    3. Third entry: Number of sinograms to average over.
      - If `None`, results in `nsinos = nslices//2`.
- `dic['rings'] (bool,float,int)`: Tuple flag for application of rings removal algorithm. 
    1. First entry: To apply (`True`) or not apply (`False`) rings filter.
    2. Second entry: Value of rings regularization parameter. Small values between `0` and `1`. **If `-1` this parameters is computed automatically**.
    3. Third entry: Value of rings block to be used, i.e., how many blocks of sinograms will be used for rings filter. Recommended `1` or `2`. Cannot be `0`.
- `dic['filter'] (str)`: Type of filter for `FDK` reconstruction.
      - Options are `'gaussian','lorentz','cosine','rectangle','hann','hamming','ramp'`
- `dic['regularization'] (float)`: Value of regularization for reconstruction (FDK or EM/TV). Small values between `0` and `1`.
      - `FDK`: Filters that uses this parameter are `'gaussian','lorentz','rectangle'`.
- `dic['iterations'] (int,int)`: Iteration numbers for `EM` reconstruction. Number of iterations and number of integration points for EM/TV.
    1. First entry: Number of iterations.
    2. Second entry: Number of integration points for each ray path.
- `dic['padding'] (int)`: Number of elements for horizontal zero-padding.
- `dic['detectorType'] (str)`: If detector type. If `'pco'` discard fist 11 rows of data. 

### Script example (version 2.2.0)

```python
import numpy as np
import h5py
import sscRaft

in_path = 'path_to_input_HDF5_file'
in_name = 'name_input_HDF5_file'

out_path = 'path_to_output_HDF5_file'
out_name = 'name_output_HDF5_file'

# Load data, flat and dark
# Mogno HDF5 example:

data = h5py.File(in_path + in_name, "r")["scan"]["detector"]["data"][:].astype(np.float32)
flat = h5py.File(in_path + in_name, "r")["scan"]["detector"]["flats"][:].astype(np.float32)
dark = h5py.File(in_path + in_name, "r")["scan"]["detector"]["darks"][:].astype(np.float32)

# Dictionary
dic = {}
dic['gpu'] = [0,1]
dic['z1[m]'] = 484*1e-3
dic['z1+z2[m]'] = 964*1e-3
dic['detectorPixel[m]'] = 1.44*1e-6
dic['reconSize'] = 2048
dic['rings'] = (True, -1, 1)
dic['normalize'] = (True, True, False)
dic['padding'] = 80
dic['shift'] = (True,0)
dic['detectorType']= 'pco'
dic['findRotationAxis'] = (500, 500, None) 
dic['filter'] = 'hann'
dic['method'] = 'fdk'
dic['regularization'] = 1
dic['fourier'] = True

recon = sscRaft.reconstruction_mogno(data, flat, dark, dic)

# Add parameter on dicionary (dic) to save as metadata on HDF5 file
dic['InputFile'] = in_path + in_name
dic['Energy[KeV]'] = 22
dic['Software'] = 'sscRaft'
dic['Version'] = sscRaft.__version__

file = h5py.File(out_path + out_name, 'w') # Save HDF5 with h5py
file.create_dataset("data", data = recon) # Save reconstruction to HDF5 output file
file.close()

file = h5py.File(out_path + out_name, 'a') # Append metadata on HDF5 output file

try:
    # Call function to save the metadata from dictionary 'dic' with the software 'sscRaft' and its version 'sscRaft.__version__'
    sscRaft.Metadata_hdf5(outputFileHDF5 = file, dic = dic, software = 'sscRaft', version = sscRaft.__version__)
except:
    print("Error! Cannot save metadata in HDF5 output file.")
    pass

file.close()

```

## Tutorial

:::caution
Be **carefull** with the `sscRaft` version!!

**Examples bellow are for version 2.2.0**
:::

First, we begin by importing the necessary python packages:

```python
    import numpy as np
    import h5py
    import sscRaft
```

- The input data type is `ndarray` from `numpy` python package. 
- Package `h5py` is used to read detector data.
- From `sscRaft` we will use the FDK reconstruction function.

Now, we set the experiment information through a dictionary:

``` python
dic = {}
dic['gpu'] = [0,1]
dic['z1[m]'] = 484*1e-3
dic['z1+z2[m]'] = 964*1e-3
dic['detectorPixel[m]'] = 1.44*1e-6
dic['reconSize'] = 2048
dic['rings'] = (True, -1, 1)
dic['normalize'] = (True, True, False)
dic['padding'] = 80
dic['shift'] = (True,0)
dic['detectorType']= 'pco'
dic['findRotationAxis'] = (500, 500, None) 
dic['filter'] = 'hann'
dic['method'] = 'fdk'
dic['regularization'] = 1
dic['fourier'] = True
```

Then, load the data:

```python
path_file = '/my_folder/path/'
name = 'sample2048.hdf5'

data = h5py.File(path_file+name, "r")["scan"]["detector"]["data"][:].astype(np.float32)
flat = h5py.File(path_file+name, "r")["scan"]["detector"]["flats"][:].astype(np.float32)
dark = h5py.File(path_file+name, "r")["scan"]["detector"]["darks"][:].astype(np.float32)
```

- `data`: Three-dimensional raw data. The axes are `[angles, slices, lenght]`.
- `flat`: Three-dimensional flat measurement. The axes are `[numberFlats, slices, lenght]`, where `numberFlats = 1` or `numberFlats = 2`.
- `dark`: Three-dimensional dark measurement. The axes are `[1, slices, lenght]`.

The function also allows for a linear interpolation of a _before_ and _after_ flat. Because of this, the `numberFlats` needs to be at most `2`, where the first entry is the _before_ flat and the second entry the _after_ flat.

:::caution
The flat and dark are only necessary for tomogram correction and are usually three-dimensional.
It can also be passed as two-dimensional entries to the reconstruction function.

**If the flat is two-dimensional, there will not be a flat interpolation!! The correction will follow with one flat.**

:::

Followiong these steps, is it possible to use the package `sscRaft` to reconstruct the aftermentioned tomogram:

```python
recon = sscRaft.reconstruction_mogno(data, flat, dark, dic)
```

- `recon (ndarray)`: Reconstructed sample object (3D). The axes are [z, y, x].

Finally, we can save the reconstructed object as desired:

```python
file = h5py.File('recon_mogno2.h5', 'w')
file.create_dataset("data", data = recon)
file.close()
```

:::note
Normalization (or correction) by _flat_ and _dark_ for transmission measurements transforms, or for phase measurements, transforms the detector raw data into a proper input for the reconstruction algorithms. The dictionary parameters are composed by 5 entries: `(A,B,C,D,E)`

- `A (bool)`: To apply or not the correction on the data
- `B (bool)`: To apply or not the logarithm on the correction. Trasmission data requires logarithm application; phase contrast measurements do not.
    
    - The mathematical operations to construct proper data for transmission measurements is

        $\   proj = -\log{ (\frac{data - dark}{flat - dark})}$

    - The mathematical operations to construct proper data for phase contrast measurements is

        $\   proj = \frac{data - dark}{flat - dark}$

- `C (int)`: The total number of frames acquired (angles quantity) - used for linear interpolation of before and after flats.
- `D (int)`: The frame number of first frame to correct (1 frame only or blocks of frames) - used for linear interpolation of before and after flats.
- `E (boll)`: After normalization, this option allows to remove negative values by addind the minimum value (on each projection)

    $\   proj = proj + | min(proj) |$

:::
