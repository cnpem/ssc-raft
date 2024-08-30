(rings)=
# Rings methods

Here you will find examples of usage of the rings filter algorithms implemented in the package `sscRaft`.

The method currently implemented on package is based on E. X. Miqueles, J. Rinkel, F. O'Dowd and J. S. V. Bermúdez 2014 paper, _Generalized Titarenko's algorithm for ring artefacts reduction_, [DOI](https://doi.org/10.1107/S1600577514016919).

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``output``: Corrected three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    output = sscRaft.rings(tomogram, dic = {'gpu': [0,1], 'regularization': -1, 'blocks':1})
```

The reference to all the input parameters can be found on the {ref}`Rings API documentation <apirings>`.
