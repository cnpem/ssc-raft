(reconstructions)=
# Reconstruction methods

Here you will find examples of usage of the reconstruction algorithms implemented in the package `sscRaft`.
The methods are divided by beam geometry: cone-beam and parallel-beam.

<!-- ## Cone-beam Geometry

The methods implemented for this geometry are:

1. Feldkamp, Davis and Kress (FDK).
2. Expectation Maximization (EM) for transmission data.

### Reconstruction Algorithm: Feldkamp, Davis and Kress (FDK) for cone-beam

The FDK Reconstruction Algorithm is a popular method for three-dimensional reconstruction from cone-beam projections.
It was developed in 1984 by Feldkamp, Davis and Kress as a practical geometry adaptation of existing analytical Filtered Backprojection strategies for reconstruction.

This reconstruction method consists of:

- Filtering conical projections.
- Backprojecting to sample reconstructions.

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``reconstruction``: Three-dimensional output reconstructed data. The axes are ``[z, y, x]`` in python syntax.

---
> **Note:** As of now, the `FDK` method does not compute individual (or block) of slices.
---

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    # Dictionary
    dic                           = {}
    dic['gpu']                    = [0,1]
    dic['angles[rad]']            = numpy.linspace(0, 2.0*numpy.pi, tomogram.shape[1])
    dic['z1[m]']                  = 484*1e-3
    dic['z1+z2[m]']               = 964*1e-3
    dic['energy[eV]']             = 22e3
    dic['detectorPixel[m]']       = 1.44*1e-6
    dic['filter']                 = 'hann'
    dic['paganin regularization'] = 0.0

    reconstruction = sscRaft.fdk(tomogram, dic)
``` -->

## Parallel-beam Geometry

The methods implemented for this geometry are:

1. Filtered Back-projection by Ray-Tracing (FBP).
2. Filtered Back-projection by Backprojection Slice Theorem (BST).
3. Expectation Maximization (EM) for transmission and emission data by Ray-Tracing.
4. Expectation Maximization (EM) for transmission on frequency domain.

### Reconstruction Algorithm: Filtered Back-projection by Ray-Tracing (FBP)

This reconstruction method consists of:

- Filtering parallel projections with Fourier Transforms.
- Backprojecting by Ray-Tracing to sample reconstructions.

The input and output data is on the following format:

- ``tomogram``

  - Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
  
- ``reconstruction``

  - Three-dimensional output reconstructed data. The axes are ``[z, y, x]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    # Dictionary
    dic                           = {}
    dic['gpu']                    = [0,1,2,3]
    dic['angles[rad]']            = numpy.linspace(0, numpy.pi, tomogram.shape[1])
    dic['filter']                 = 'lorentz'
    dic['paganin regularization'] = 0.0 
    dic['padding']                = 4 
    dic['blocksize']              = 0 

    reconstruction = sscRaft.fbp(tomogram, dic)
```

The reference to all the input parameters can be found on the {ref}`FBP API documentation <apifbp>`.

### Reconstruction Algorithm: Filtered Back-projection by Backprojection Slice Theorem (BST) for parallel-beam

This reconstruction method consists of:

- Filtering parallel projections with Fourier Transforms.
- Backprojecting by Slice Theorem to sample reconstructions.

The input and output data is on the following format:

- ``tomogram``

  - Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

- ``reconstruction``

  - Three-dimensional output reconstructed data. The axes are ``[z, y, x]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    # Dictionary
    dic                           = {}
    dic['gpu']                    = [0,1,2,3]
    dic['angles[rad]']            = numpy.linspace(0, numpy.pi, tomogram.shape[1])
    dic['filter']                 = 'lorentz'
    dic['paganin regularization'] = 0.0 
    dic['padding']                = 4 
    dic['blocksize']              = 0 

    reconstruction = sscRaft.bst(tomogram, dic)
```

The reference to all the input parameters can be found on the {ref}`BST API documentation <apibst>`.

### Reconstruction Algorithm: Expectation Maximization (EM) for parallel-beam Transmission and Emission data

There are three EM methods implemented

1. Expectation Maximization by Ray Tracing for parallel-beam Transmission `'tEMRT'`
2. Expectation Maximization by Ray Tracing for parallel-beam Emission `'eEMRT'`
3. Expectation Maximization on frequency domain for parallel-beam Transmission `'tEMFQ'`

---
> **Caution:** Emission EM by Ray Tracing considers the input data as a tomogram/sinogram {math}`T`. The transmission EM methods consider the input data
> as photon-counts {math}`C`. One can simulate photon-counts from the tomogram by the relation {math}`C = F \exp(-T)`, where {math}`F` is the flat measurement.
---

The input and output data for the emission EM methods is on the following format:

- ``tomogram``

  - Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

- ``reconstruction``

  - Three-dimensional output reconstructed data. The axes are ``[z, y, x]`` in python syntax.

The input and output data the transmission EM methods is on the following format:

- ``counts``

  - Three-dimensional raw photon-counts data. The axes are ``[slices, angles, rays]`` in python syntax.

- ``flat``

  - Three-dimensional flat measurement. The axes are ``[slices, numberFlats, rays]`` in python syntax. Flat data is a dictionary parameter of the EM function.

- Example for Expectation Maximization by Ray Tracing for parallel-beam Emission

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    # Dictionary
    dic                           = {}
    dic['gpu']                    = [0,1,2,3]
    dic['angles[rad]']            = numpy.linspace(0, numpy.pi, tomogram.shape[1])
    dic['padding']                = 2 
    dic['blocksize']              = 0 
    dic['iterations']             = 10
    dic['method']                 = 'eEMRT'

    reconstruction = sscRaft.em(tomogram, dic)
```

- Example for Expectation Maximization by Ray Tracing for parallel-beam Transmission

```python
    import numpy
    import sscRaft

    '''Load data-set
    counts = ...
    flat   = ...
    '''

    # Dictionary
    dic                           = {}
    dic['gpu']                    = [0,1,2,3]
    dic['angles[rad]']            = numpy.linspace(0, numpy.pi, tomogram.shape[1])
    dic['flat']                   = flat
    dic['padding']                = 2 
    dic['blocksize']              = 0 
    dic['iterations']             = 10
    dic['method']                 = 'tEMRT'

    reconstruction = sscRaft.em(counts, dic)
```

- Example for Expectation Maximization on frequency domain for parallel-beam Transmission

```python
    import numpy
    import sscRaft

    '''Load data-set
    counts = ...
    flat   = ...
    '''

    # Dictionary
    dic                           = {}
    dic['gpu']                    = [0,1]
    dic['angles[rad]']            = numpy.linspace(0, numpy.pi, tomogram.shape[1])
    dic['flat']                   = flat
    dic['padding']                = 2 
    dic['blocksize']              = 0 
    dic['iterations']             = 10
    dic['method']                 = 'tEMFQ'

    reconstruction = sscRaft.em(counts, dic)
```

The reference to all the input parameters can be found on the {ref}`EM API documentation <apiem>`.
