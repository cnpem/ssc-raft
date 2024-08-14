from ..rafttypes import *

def transpose(data):
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

    libraft.transpose_cpu(data_ptr,
            ctypes.c_int(sizex),
            ctypes.c_int(sizey),
            ctypes.c_int(sizez))

    data = data.reshape((sizey, sizez, sizex))

    return data

def transpose_np(data):
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
