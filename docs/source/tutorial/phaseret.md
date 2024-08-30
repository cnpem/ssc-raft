(phaseret)=
# Phase Retrieval methods

Here you will find examples of usage of phase retrieval algorithms implemented in the package `sscRaft`.

Application of phase retrieval methods based on the Transport of Equation (TIE) approach.
Currently, the method implemented is based on D. Paganin, S. C. Mayo, T. E. Gureyev, P. R. Miller, S. W. Wilkins 2002 paper, _Simultaneous phase and amplitude extraction from a single defocused image of a homogeneous object_, [DOI](https://doi.org/10.1046/j.1365-2818.2002.01010.x).

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``output``: Corrected three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

---
> **Note:** The data input needs to be corrected by flat (or empty) and dark before the method.
---

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    output = sscRaft.rings(tomogram, dic = {'gpu': [0,1], 'method': 'paganin', 'beta/delta': 1e-3,
                                            'detectorPixel[m]': 3.61e-6, 'z2[m]':500e-3, 
                                            'energy[eV]': 22e3, 'magn': 1.1,
                                            'blocksize': 0})
```

The reference to all the input parameters can be found on the {ref}`Phase Retrieval API documentation <apiphaseretrieval>`.
