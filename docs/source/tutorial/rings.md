(rings)=
# Rings methods

Ring artifacts are one of the most ubiquitous types of artifacts present at tomographic reconstructions
and can present quite a challenge when it comes to analyzing and segmenting tomographic images. 
Here you will find examples of usage of the algorithms implemented in the package `sscRaft`.

## Rings artifacts

Rings artifacts are exactly what the name suggests: they appear as rings at the reconstruction level, as shown in Figure 1. 
They occur when the measured data present stripes artifacts in its acquisition. These stripes have different categories and are thoroughly discussed at the work {cite}`Vo18`, just as in the [sarepy](https://sarepy.readthedocs.io/toc/section2.html) documentation pages.

| ![img](../../static/images/stripes_to_rings_demo_2.png) | 
| :-------: | 
| Figure 1: **(a)** Sinogram of a demonstration phantom; **(b)** sinogram in (a) corrupted by vertical stripes; **(c)** reconstruction of sinogram in (a); **(d)** reconstruction of sinogram in (b).| 

## All Stripes Method - Parameters and Usage

The ``ssc-raft`` presents a ``multi-GPU`` implementation of the _all stripes_ algorithm introduced in {cite}`Vo18`, that applies a sequence of processing stages that act on some specific categories of stripes in order to mitigate them. Details can be found in {cite}`macul2024rings`. 

### Usage

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``output``: Corrected three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    output = sscRaft.rings_remove_all_stripes(tomo, ngpus = 1, settings = { 'size_median': small_to_median_stripes_size,
                                                                            'large_stripes_size': large_stripes_size,
                                                                            'background_ratio': background_ration,
                                                                            'drop_ratio': drop_ratio,
                                                                            'dim':1})
```

The reference to all the input parameters can be found on the {ref}`All Stripes API documentation <apiallstripes>`.

(paramAllS)=
### Parameters selection 

- **``background_ratio (default:10.0)``** - in one of the steps of the "remove all stripes" algorithm (referred to as algorithm 4 at {cite}`Vo18`), a segmentation
stage takes place as a means to separate large stripes from the image's background. In this process, this parameter can be viewed as a form to adjust the
sensitivity of the segmentation algorithm in its search for large stripes at the sinogram, setting smaller values makes the algorithm more sensitive.
Recommended values are in the range from _1.1_ to _3_ as of [sarepy documentation](https://sarepy.readthedocs.io/toc/section3_1/section3_1_6.html), smaller values
might degrade the sinogram depending on the kind of stripes at it, however, values bigger than 3 tend not to result in significant improvements.

- **``size_median (default:21)``** - defines the window size of the median filter used by the algorithm to attenuate small to medium stripes. It's recommended to keep
its value in the range of _21_ to _40_ for most of the situations. For large values (above 90), it is possible that execution errors (CUDA errors from illegal
memory access) show up. It's also worth to notice that there's a relation between the size of this parameter and the
execution time of this ring removal method, the larger it is, the longer it takes to finish. Values too small (less than 15) tend to degrade sinograms
(and thus reconstructions) considerably.

- **``large_stripes_size (default:61)``** - defines the window size of the median filter used to attenuate large stripes. It's recommended to keep its values inside the
range of 60 to 121 for most situations. Just as with the ``size_median`` parameters, choosing values too large might incur in execution errors. Also, the relation between
the size of this parameter and the execution time of this method remains valid.

- **``dim (default:1)``** - dimension of the windows used at the median filters. This implementation only supports 1D filters (``dim=1``).

- **``drop_ratio (default:0.1)``** - percentage of pixels to be dropped out at the top and at the bottom of the
processed sinogram in the step 2 of algorithm 5 of {cite}`Vo18`, this parameter might be used to reduce the possibility of
false stripes detection, which might occur due to high-frequency edges of the sinogram. For most of the cases, it's recommended to keep
this value around _0.8_ but values between _0.1_ and _0.8_ are acceptable. Values above 0.8 are clipped to 0.8 internally and
the same is valid for values below 0.

## All Stripes Multiaxis Method - Parameters and Usage

This method is a twist of the original _all stripes_ method, where the _all stripes_ method is applied to the tomogram in another axis (``axis 2``) to mitigate horizontal stripes. 
The input and parameters are the same as the ``rings_remove_all_stripes()`` function.                 

### Usage

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``output``: Corrected three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    output = sscRaft.rings_remove_all_stripes_multiaxis(tomo, ngpus = 1, settings = {   'size_median': small_to_median_stripes_size,
                                                                                        'large_stripes_size': large_stripes_size,
                                                                                        'background_ratio': background_ration,
                                                                                        'drop_ratio': drop_ratio,
                                                                                        'dim':1})
```

The reference to all the input parameters can be found on the {ref}`All Stripes Multiaxis API documentation <apiallstripesM>`.

### Parameters selection

Refer to _all stripes_ {ref}`paramAllS`.

## Titarenko Method - Parameters and Usage

This method is based on the work of E.X. Miqueles et al (2014) {cite}`miq2014Titarenko`, developed as a generalization of {cite}`titarenko2010a`.

### Usage

The input and output data is on the following format:

- ``tomogram``: Three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.
- ``output``: Corrected three-dimensional tomogram data. The axes are ``[slices, angles, rays]`` in python syntax.

```python
    import numpy
    import sscRaft

    '''Load data-set
    tomogram = ...
    '''

    output = sscRaft.titarenko_rings(tomogram, dic = {'gpu': [0,1], 'regularization': -1, 'blocks':1})
```

The reference to all the input parameters can be found on the {ref}`Titarenko API documentation <apititarenko>`.

### Parameters selection

- **``regularization (default:-1)``** - Titarenko's regularization value. Values between {math}`[0,1]`. The value -1 compute this parameter automatically.

- **``blocks (default:1)``** - Blocks of sinogram columns to be used. Even values between {math}`[1,20]`.


## References

```{bibliography}

