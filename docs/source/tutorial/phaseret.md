(phaseret)=
# Phase Retrieval methods

Here you will find examples of usage of phase retrieval algorithms implemented in the package `sscRaft`.

Application of phase retrieval methods based on the Transport of Equation (TIE) approach.
Currently, the method implemented is based on 

## Paganin Filter

This method is an implementation of the classical Paganin filter introduced in {cite}`pag2002`.

### Usage

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``output``: Corrected three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

---
> **Note:** The measurement needs to be corrected by flat (or empty) and dark previously.
---

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    output = sscRaft.phase_retrieval(tomogram, dic = {'gpu': [0,1], 'method': 'paganin', 'beta/delta': 1e-3,
                                                      'detectorPixel[m]': 3.61e-6, 'z2[m]':500e-3, 
                                                      'energy[eV]': 22e3, 'magn': 1.1})
```

For the classic Paganin method, we have to apply {math}`-\log()` on the output using the dictionary entry ``\'post_process\': True``

```python
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    output = sscRaft.phase_retrieval(tomogram, dic = {'gpu': [0,1], 'method': 'paganin', 'beta/delta': 1e-3,
                                                      'detectorPixel[m]': 3.61e-6, 'z2[m]':500e-3, 
                                                      'energy[eV]': 22e3, 'magn': 1.1, 'post_process': True})
```

The reference to all the input parameters can be found on the {ref}`Phase Retrieval API documentation <apiphaseretrieval>`.
