(excentric)=
# Excentric tomography

Here you will find examples of usage of algorithms for excentric tomography implemented in the package `sscRaft`.

## Finding the offset

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``offset``: Integer value that corresponds to the excentric tomography offset. It already takes into account the rotation axis deviation.

The corrected offset can be found using just a few slices, with a minimum of one slice.


---
> **Note:** The measurement needs to be corrected by flat (or empty) and dark previously.
---

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    offset = sscRaft.getOffsetExcentricTomo(tomogram, gpus = [0])
```

## Stitching

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``tomogram``: Three-dimensional stitched tomogram data. The axes are ``[slices, angles / 2, 2 * rays]`` in python syntax.

The resulting stitched tomogram have 180 degrees. 

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    tomogram = sscRaft.stitchExcentricTomo(tomogram, offset, gpus = [0])
```

The offset value can be found through the function ``getOffsetExcentricTomo()``.

The reference to all the input parameters can be found on the {ref}`Alignment API documentation <apistitch>`.
