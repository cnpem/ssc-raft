(pipelines)=
# Tomography pipelines

Here you will find examples of usage of tomography pipelines implemented in the package ``sscRaft``.

## Parallel geometry C pipeline

### Usage

The input and output data is on the following format:

- ``data``: Three-dimensional tomogram data. The axes are ``[slices, angles, length]`` in python syntax.
- ``flat``: Three-dimensional flat measurement. The axes are ``[slices, numberFlats, rays]`` in python syntax, where ``numberFlats = 1`` or ``numberFlats = 2``.
- ``dark``: Three-dimensional dark measurement. The axes are ``[slices, 1, length]`` or ``[slices, length]`` in python syntax.
- ``angles``: List of angles in radians. Needs to have the same size of ``data`` second axes.

```python
    import numpy
    import sscRaft

    '''Load data-set
    data = ...
    flat = ...
    dark = ...
    '''

    offset = sscRaft.FastReconPipeline(data, flat, dark, angles = angles, gpus = [0], dic = {'reconstruction method': 'fbp',
                                                                                            'rings method': 'all stripes',
                                                                                            'rotation axis method': 'rot axis recon',
                                                                                            'filter': 'hamming',
                                                                                            'beta/delta': 0.0,
                                                                                            'detectorPixel[m]': 1.44e-6, 
                                                                                            'z2[m]':1.0e-3,
                                                                                            'energy[eV]':22e3,
                                                                                            'padding': 0.25,
                                                                                            'padd_mode': 'zero',
                                                                                            'zoom padding': 0.0,
                                                                                            'zoom padd_mode': 'edge',
                                                                                            'rotation axis offset': None,
                                                                                            'excentric offset': 0,
                                                                                            'blocksize': 16,
                                                                                            'rings size_median': 21,
                                                                                            'rings large_stripes_size': 61,
                                                                                            'rings background_ratio': 10.0,
                                                                                            'rings drop_ratio': 0.1,
                                                                                            'rings dim': 1})
```
For padding usage, we refer to [Padding documentation](padding.md).The reference to all the input parameters can be found on the {ref}`API documentation <apipipeA>`.

