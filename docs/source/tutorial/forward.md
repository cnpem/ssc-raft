(forward)=
# Forward methods

Here you will find examples of usage of the forward operators implemented in the package `sscRaft`.
The methods are divided by beam geometry: cone-beam and parallel-beam.

All methods discussed here, unless stated, returns projection with physical units in SI, i.e. in meter ({math}`m`), considering the pixel size in the ``pixel`` input argument. 
If no pixel size is passed, the algorithm consider ``pixel = 1.0`` by default.

## Parallel-beam geometry

The methods implemented for this geometry are:

1. Radon Transform by Ray-Tracing.

### Radon Transform by Ray-Tracing

#### Usage

The input and output data is on the following format:

- ``phantom``: Two- or three-dimensional volume data of a sample. The axes are ``[z, y, x]`` in python syntax.
  
- ``tomogram``: Two- or three-dimensional output tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    phantom = ...
    '''
    nangles  = phantom.shape[1]
    angles   = numpy.linspace(0, numpy.pi, nangles)

    tomogram = sscRaft.radon_RT(phantom, angles, gpus = [0,1], pixel = 1.0)
```

The reference to all the input parameters can be found on the {ref}`Radon API documentation <apiradonA>`.

## Cone-beam geometry

The methods implemented for this geometry are:

1. Conebeam geometric adaptation of Radon Transform by Ray-Tracing.

### Cone Radon Transform by Ray-Tracing

#### Usage

The input and output data is on the following format:

- ``phantom``: Three-dimensional volume data of a sample. The axes are ``[z, y, x]`` in python syntax.
  
- ``tomogram``: Three-dimensional output tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    phantom = ...
    '''

    tomogram = sscRaft.cbradon(phantom, ngpus = 1, dic = { 'z1[m]': 1.0, 'z1+z2[m]': 2.0, 'number of projections': 64, 
                                                           'phantomPixel[m]': 1.0, 'detectorPixel[m]': 1.0 } )
```

The reference to all the input parameters can be found on the {ref}`Radon API documentation <apiradonB>`.