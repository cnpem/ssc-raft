(excentric)=
# Eccentric Tomography

This section provides examples of the eccentric tomography algorithms available in `sscRaft`.

## Finding the Offset

### Usage

The input and output data are:

- `data`: Three-dimensional tomographic data with axes ordered as `[slices, angles, rays]`.
- `offset`: Integer value corresponding to the eccentric tomography offset. The returned value already accounts for the rotation-axis displacement.

The offset can be estimated using only a small subset of slices, and a single slice is often sufficient. Since the algorithm runs on a single GPU, it is recommended to use the minimum number of slices required to reduce memory consumption.

> 📝 **Note**
>
> The projection data must be corrected with flat-field (or empty-beam) and dark-field images before estimating the offset.

```python
import sscRaft

# Load a subset of slices
data = data[slice_init:slice_end]

offset = sscRaft.getOffsetExcentricTomo(data, gpus=[0])
```

See the {ref}`API documentation <apistitchA>` for a complete description of all input parameters.

## Stitching

### Usage

The input and output data are:

- `data`: Three-dimensional tomographic data with axes ordered as `[slices, angles, rays]`.
- `output`: Three-dimensional stitched tomographic data with axes ordered as `[slices, angles // 2, 2 * rays]`.

The resulting stitched dataset spans 180 degrees and is automatically centered, with the rotation-axis offset corrected during the stitching process.

```python
import sscRaft

# Load dataset
# data = ...

output = sscRaft.stitchExcentricTomo(data, offset, gpus=[0])
```

The offset parameter can be obtained using the `getOffsetExcentricTomo()` function described above.

See the {ref}`API documentation <apistitchB>` for a complete description of all input parameters.
