(rotaxis)=
# Rotation Axis methods

Here you will find examples of usage of algorithms implemented in the package `sscRaft` for
alignment of the tomogram before reconstruction.

## Finding the offset

### Parallel beam method

The input and output data is on the following format:

- ``frame0``: Two-dimensional frame corresponding to the 0 degree raw data. The axes are ``[slices, rays]`` in python syntax.
- ``frame1``: Two-dimensional frame corresponding to the 180 degree raw data. The axes are ``[slices, rays]`` in python syntax.
- ``offset``: Integer value that corresponds to the rotational axis offset.


```python
    import numpy
    import sscRaft

    '''Load data-set
    frame0 = frame corresponding to the 0 degree 
    frame1 = frame corresponding to the 180 degree
    flat   = ...
    dark   = ...
    '''

    offset = sscRaft.Centersino(frame0, frame1, flat, dark)
```

### Conical beam method

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``offset``: Integer value that corresponds to the rotational axis offset. The sign is opposit to the one computed with ``Centersino()``.

---
> **Note:** The measurement needs to be corrected by flat (or empty) and dark previously.
---

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ... 
    '''
    offset = sscRaft.find_rotation_axis_360(tomogram, nx_search=500, nx_window=500, nsinos=None)
```

## Rotational axis correction

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``tomogram``: Three-dimensional stitched tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    tomogram = sscRaft.correct_rotation_axis(tomogram, offset, gpus = [0])
```

The offset value can be found through the function ``Centersino()``.

The reference to all the input parameters can be found on the {ref}`Rotational axis API documentation <apirotaxis>`.
