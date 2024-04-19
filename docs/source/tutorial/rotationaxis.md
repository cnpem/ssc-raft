---
title: Rotation Axis
description: Rotation axis deviation correction with ssc-raft
sidebar_position: 2
---

:::note
For documentation on functions see [sscRaft documentation](https://gcc.lnls.br/ssc/ssc-raft/index.html).
:::

## Rotation axis correction (version 2.2.0)

Correction of rotation axis deviation.

It is a `CPU` (`Python`) function and searches for the rotation axis index of a sample measured on an angle range of more than 180 degrees. It minimizes the symmetry error between two projections on equivalent angles.

There is two functions: one returns **only** how many pixels the rotation axis is deviated from the center, the other returns the same index value and **also** returns the data with the rotation axis corrected.

:::caution
This function works better with a data input that is already normalized by _flat_ and _dark_, and _before_ rings filter application.
:::


### Function call (version 2.2.0)

Returns deviation index:

```python
import sscRaft
'''set arrays:

tomogram = ...
'''

deviation = sscRaft.find_rotation_axis_360(tomogram, nx_search, nx_window, nsinos)
```

Returns deviation index and corrected data:

```python
import sscRaft
'''set dictionary and arrays:

dic = {}
...

tomogram = ...
'''

tomogram, deviation = sscRaft.correct_rotation_axis360(tomogram, dic)
``` 

### Dictionary and function parameters (version 2.2.0)

Returns deviation index:

- `nx_search`:  Value of width of the search
- `nx_window`: Value of sinogram size will be used in the horizontal axis.
- `nsinos`: Number of sinograms to average over.
    * If `None`, results in `nsinos = nslices//2`.

Returns deviation index and corrected data:

- `dic['shift'] (bool,int)`: Tuple. Rotation axis automatic corrrection.
    1. First entry: To apply (`True`) or not apply (`False`) **automatic** rotation axis deviation correction. 
    2. Second entry: Value of rotation axis deviation if it is already know.  
- `dic['findRotationAxis'] (int,int,int)`: Extra parameters for rotation axis function. 
    1. First entry:  Value of width of the search (`nx_search`).
    2. Second entry: Value of sinogram size will be used in the horizontal axis (`nx_window`).
    3. Third entry: Number of sinograms to average over (`nsinos`).
        * If `None`, results in `nsinos = nslices//2`.
- `dic['padding'] (int)`: Number of elements for horizontal zero-padding. 

:::note
The `numpy ndarray` input `tomogram` for both functions needs to have dimensions `[slices, angles, rays]`.
The corrected data `tomogram` returned have dimensions `[slices, angles, rays]`.
:::

### Script example (version 2.2.0)

Returns deviation index:

```python
import numpy as np
import h5py
import sscRaft

# Load or compute tomogram
# tomogram = ...

nx_search = 500
nx_window = 500
nsinos = None

deviation = sscRaft.find_rotation_axis_360(tomogram, nx_search, nx_window, nsinos)

```

Returns deviation index and corrected data:

```python
import numpy as np
import h5py
import sscRaft

# Load or compute tomogram
# tomogram = ...

# Dictionary
dic = {}
dic['padding'] = 0
dic['shift'] = (True,0)
dic['findRotationAxis'] = (500, 500, None) 

tomogram, deviation = sscRaft.correct_rotation_axis360(tomogram, dic)

```
