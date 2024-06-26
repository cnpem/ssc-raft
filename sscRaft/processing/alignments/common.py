from ...rafttypes import *

def shift_slices_via_fourier_transform_parallel(data3D, shift_y, shift_x,fill_in=None):
    """
    Shifts slices of a 3D array in the x and y directions using Fourier transforms.

    Parameters:
    -----------
    data3D : numpy.ndarray
        The 3D input data array of shape (N, Y, X).
    shift_y : numpy.ndarray
        An array of shape (N, 1) containing the shift values for the y-axis.
    shift_x : numpy.ndarray
        An array of shape (N, 1) containing the shift values for the x-axis.
    fill_in : float or None, optional
        The value to fill in the regions created by the shift. If None, no filling is applied.

    Returns:
    --------
    numpy.ndarray
        The shifted 3D array with the same shape as the input data.

    Notes:
    ------
    This function performs the following steps:
    1. Computes the 2D Fourier transform of each slice in the 3D array.
    2. Applies a phase shift in the Fourier domain to achieve the desired spatial shift.
    3. Computes the inverse Fourier transform to get the shifted slices in the spatial domain.
    4. Optionally fills in the regions created by the shift with a specified value.
    
    The function uses parallel processing to speed up the computation.

    Example:
    --------
    >>> N, Y, X = 10, 256, 256
    >>> data3D = np.random.rand(N, Y, X)
    >>> shift_y = np.random.rand(N, 1)
    >>> shift_x = np.random.rand(N, 1)
    >>> shifted_result = shift_slices_via_fourier_transform_parallel(data3D, shift_y, shift_x, fill_in=0)
    >>> print(shifted_result)
    """

    N, Y, X = data3D.shape
    ft_data = np.fft.fftshift(np.fft.fft2(data3D), axes=(1, 2))
    ft_freq_y = np.fft.fftshift(np.fft.fftfreq(Y, d=1)) # shift in pixels, hence d=1
    ft_freq_x = np.fft.fftshift(np.fft.fftfreq(X, d=1))
    XX, YY = np.meshgrid(ft_freq_x, ft_freq_y)

    def process_slice(i):
        exponential_part = np.exp(shift_y[i] * 1j * 2 * np.pi * YY  + shift_x[i] * 1j * 2 * np.pi * XX )
        shifted_slice = np.fft.ifft2(np.fft.ifftshift(ft_data[i] * exponential_part))
        
        if fill_in is not None:
            if shift_y[i] > 0:
                shifted_slice[-int(np.ceil(shift_y[i])):,:] = fill_in # np.ceil not to round to zero
            if shift_y[i] < 0:
                shifted_slice[0:-int(np.ceil(shift_y[i])),:] = fill_in
            if shift_x[i] > 0:
                shifted_slice[:,-int(np.ceil(shift_x[i])):] = fill_in
            if shift_x[i] < 0:
                shifted_slice[:,0:-int(np.ceil(shift_x[i]))] = fill_in
        
        return shifted_slice


    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_slice, range(N)), desc=f"Shifting slices using {executor._max_workers} workers"))

    return np.array(results)


def find_common_non_null_area(data):
    masked_volume = np.where(data==0,0,1)
    product = np.prod(masked_volume,axis=0)
    data = data[:]*product
    return data

def remove_zeroed_borders(volume):
    """
    Remove the null borders of a volume of images 
    """
    
    not_null = np.argwhere(volume[0])
    # Bounding box of non-0 pixels.
    x0, y0 = not_null.min(axis=0)
    x1, y1 = not_null.max(axis=0) + 1   # slices are exclusive at the top
    # Get the contents of the bounding box.
    return volume[:,x0:x1, y0:y1]
