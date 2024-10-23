from ..rafttypes import *

def transpose(data, axes=(0,1,2)):
    """CPU float function to apply transpose on a 3D array.

    Args:
        data (ndarray): data. Axes are [z,y,x].

    Returns:
        (ndarray): data. Axes are [z,y,x].
    """
    sizex = data.shape[-1]
    sizey = data.shape[-2]

    if len(data.shape) == 2:
        sizez = 1
    else:
        sizez = data.shape[0]

    data     = CNICE(data)
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)

    if axes == (1, 0, 2):
        data = _transpose_cpu_zyx2yzx(data)
    elif axes == (2, 1, 0):
        libraft.transpose_cpu_zyx2xyz(data_ptr,
            ctypes.c_int(sizex),
            ctypes.c_int(sizey),
            ctypes.c_int(sizez))
    else:
        raise ValueError("Transpose {} not implemented".format(axes))

    old_shape = (sizez, sizey, sizex)
    new_shape = tuple(old_shape[i] for i in axes)

    data = data.reshape(new_shape)

    return data

def _transpose_cpu_zyx2yzx(data):
    # import pdb; pdb.set_trace()
    temp = numpy.swapaxes(data, 0, 1)
    data = data.reshape(temp.shape)
    data[...] = temp
    return data

def flip_x(data):
    """CPU float function to flip x axis inplace.

    Args:
        data (ndarray): data [z,y,x].

    Returns:
        (ndarray): data flipped values in x axis.
    """
    sizex = data.shape[-1]
    sizey = data.shape[-2]

    if len(data.shape) == 2:
        sizez = 1
    else:
        sizez = data.shape[0]

    data     = CNICE(data)
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)

    libraft.flip_x(data_ptr,
            ctypes.c_int(sizex),
            ctypes.c_int(sizey),
            ctypes.c_int(sizez))

    return data

def flip_x_np(data):
    data[..., ::-1] = data
    return data
