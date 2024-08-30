(reconstructions)=
# Reconstruction methods

Here you will find examples of usage of the reconstruction algorithms implemented in the package `sscRaft`.
The methods are divided by beam geometry: cone-beam and parallel-beam.

## Cone-beam Geometry

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

    angles = numpy.linspace(0, 2.0*numpy.pi, tomogram.shape[1])

    reconstruction = sscRaft.fdk(tomogram, dic = {'gpu': [0,1], 'angles[rad]': angles, 'beta/delta': 0.0,
                                                  'detectorPixel[m]': 3.61e-6, 'z1[m]':1000e-3, 'z1+z2[m]':2000e-3, 'z2[m]':500e-3, 
                                                  'energy[eV]': 22e3, 'filter': 'hamming', 
                                                  'padding': 2, 'blocksize': 0})
```

The reference to all the input parameters can be found on the {ref}`FDK API documentation <apifdk>`.

## Parallel-beam Geometry

The methods implemented for this geometry are:

1. Filtered Back-projection by Ray-Tracing (FBP) or by Backprojection Slice Theorem (BST).
2. Expectation Maximization (EM) for transmission and emission data by Ray-Tracing.
3. Expectation Maximization (EM) for transmission on frequency domain.

### Reconstruction Algorithm: Filtered Back-projection

This reconstruction method consists of filtering parallel projections with Fourier Transforms
and backprojecting by two methods:

- Backprojecting by Ray-Tracing to sample reconstructions.
- Backprojecting by Backprojection Slice Theorem to sample reconstructions.

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

    angles = numpy.linspace(0, numpy.pi, tomogram.shape[1])

    reconstruction = sscRaft.fbp(tomogram, dic = {'gpu': [0,1], 'method': 'RT', 'angles[rad]': angles, 
                                                  'beta/delta': 0.0, 'detectorPixel[m]': 3.61e-6, 
                                                  'z1[m]': 1000e-3, 'z1+z2[m]':2000e-3, 'z2[m]':500e-3, 
                                                  'energy[eV]': 22e3, 'filter': 'hamming', 
                                                  'padding': 2, 'blocksize': 0})
```

The reference to all the input parameters can be found on the {ref}`FBP API documentation <apifbp>`.

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

    angles = numpy.linspace(0, numpy.pi, tomogram.shape[1])

    reconstruction = sscRaft.em(tomogram, dic = {'gpu': [0,1], 'method': 'eEMRT', 'angles[rad]': angles, 
                                                  'iterations': 10, 'padding': 2, 'blocksize': 0})
```

- Example for Expectation Maximization by Ray Tracing for parallel-beam Transmission

```python
    import numpy
    import sscRaft

    '''Load data-set
    counts = ...
    flat   = ...
    '''

    angles = numpy.linspace(0, numpy.pi, tomogram.shape[1])

    reconstruction = sscRaft.em(counts, dic = {'gpu': [0,1], 'method': 'tEMRT', 
                                               'angles[rad]': angles, 'flat':flat,
                                               'iterations': 10, 'padding': 2, 'blocksize': 0})
```

- Example for Expectation Maximization on frequency domain for parallel-beam Transmission

```python
    import numpy
    import sscRaft

    '''Load data-set
    counts = ...
    flat   = ...
    '''

    angles = numpy.linspace(0, numpy.pi, tomogram.shape[1])

    reconstruction = sscRaft.em(counts, dic = {'gpu': [0,1], 'method': 'tEMFQ', 
                                               'angles[rad]': angles, 'flat':flat, 'detectorPixel[m]': 3.61e-6,
                                               'iterations': 10, 'padding': 2, 'blocksize': 0})
```

The reference to all the input parameters can be found on the {ref}`EM API documentation <apiem>`.
