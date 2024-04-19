(forward)=
# Forward methods

Here you will find examples of usage of the forward operators implemented in the package `sscRaft`.
The methods are divided by beam geometry: cone-beam and parallel-beam.

## Parallel-beam Geometry

The methods implemented for this geometry are:

1. Radon Transform by Ray-Tracing.

### Forward Algorithm: Radon Transform by Ray-Tracing

The input and output data is on the following format:

- ``phantom``

  - Three-dimensional volume data of a sample. The axes are ``[z, y, x]`` in python syntax.
  
- ``tomogram``

  - Three-dimensional output tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    phantom = ...
    '''

    nangles  = phantom.shape[1]
    angles   = numpy.linspace(0, numpy.pi, nangles)

    tomogram = sscRaft.radon_RT(phantom, angles, gpus = [0,1])
```

The reference to all the input parameters can be found on the {ref}`Radon API documentation <apiradon>`.
