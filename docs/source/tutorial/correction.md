(correction)=
# Background correction

Background correction by _flat_, denoted by {math}`D_f`, and _dark_, denoted by {math}`D_d`, for transmission or for phase contrast measurements,
transforms the detector raw data into a proper input for the reconstruction algorithms.

The mathematical operations to construct proper data, denoted by {math}`D`, for transmission measurements is

```{math}
    T = - \log{(\frac{D - D_d}{D_f - D_d})}.
```

The mathematical operations to construct proper data for phase contrast measurements is

```{math}
T = \frac{D - D_d}{D_f - D_d},
```

where {math}`T` is the corrected tomogram.

The input and output data is on the following format:

* {math}`D` - ``data``

  * Three-dimensional raw data. The axes are ``[slices, angles, rays]`` in python syntax.

* {math}`T` - ``tomogram``

  * Three-dimensional output corrected data. The axes are ``[slices, angles, rays]`` in python syntax.

* {math}`D_f` - ``flat``

  * Three-dimensional flat measurement. The axes are ``[slices, numberFlats, rays]`` in python syntax, where ``numberFlats = 1`` or ``numberFlats = 2``.

* {math}`D_d` - ``dark``

  * Three-dimensional dark measurement. The axes are ``[slices, 1, rays]`` in python syntax.

The function also allows for a linear interpolation of a _before_ and _after_ flat. Because of this, the ``numberFlats`` needs to be at most 2, where the first entry is
the _before_ flat, taken before the begining of data acquisition, and the second entry the _after_ flat, taken after data acquisition.

---
> **Caution:** The flat and dark are only necessary for tomogram correction and are usually three-dimensional.
> It can also be passed as two-dimensional entries to the reconstruction function.
> *If the flat is two-dimensional, there will not be a flat interpolation! The correction will follow with one flat.*
---

Script example:

```python
    import sscRaft

    '''Load data-set
    data = ...
    flat = ...
    dark = ...
    '''
    
    tomogram = sscRaft.correct_background(data, flat, dark, gpus = [0], is_log = True)
```

The reference to all the input parameters can be found on the API documentation {ref}`apicorrection`.
