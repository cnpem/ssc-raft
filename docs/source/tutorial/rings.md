---
title: Rings
description: Rings filter with ssc-raft
sidebar_position: 3
---

:::note
For documentation on functions see [sscRaft documentation](https://gcc.lnls.br/ssc/ssc-raft/index.html).
:::

## Tomogram rings filtering (version 2.2.0)

Filter for tomogram rings removal. The method implemented here is based on E. X. Miqueles, J. Rinkel, F. O'Dowd and J. S. V. Bermúdez 2014 paper, [_Generalized Titarenko's algorithm for ring artefacts reduction_](https://doi.org/10.1107/S1600577514016919).

### Function call (version 2.2.0)

```python
import sscRaft
'''set dictionary and arrays:

dic = {}
...

tomogram = ...
'''

recon = sscRaft.rings(tomogram, dic)
``` 

### Dictionary parameters (version 2.2.0)

- `dic['gpu'] (ndarray)`: List of gpus for processing. 
- `dic['lambda rings'] (float)`: Regularization of rings. Small values between `0` and `1`.
    * If `dic['lambda rings'] < 0`, the regularization value is computed automatically.
- `dic['rings block'] (int)`: Block of slices to be used in rings filter.
    * For cone-beam geometry is recommended, for now, use `dic['rings block'] = 2`.

:::note
The `numpy ndarray` input `tomogram` for the `sscRaft.rings()` function needs to have dimensions `[slices, angles, rays]`.
The corrected data `tomogram` returned have dimensions `[slices, angles, rays]`.
:::

### Script example (version 2.2.0)

Consider this Mogno beamline data example: 

```python
import numpy as np
import h5py
import sscRaft

# Load or compute tomogram
# tomogram = ...

# Dictionary
dic = {}
dic['gpu'] = [0,1] 
dic['lambda rings'] = -1 
dic['rings block'] = 1

tomogram = sscRaft.rings(tomogram, dic)

```


