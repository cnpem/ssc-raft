from ...rafttypes import *

from .common import shift_slices_via_fourier_transform_parallel, find_common_non_null_area, remove_zeroed_borders

### Cross Correlation ### 
def alignment_cross_correlation(data,angles, max_downscaling_factor=32, fft_upsampling=100, remove_null_borders = True, use_gradient = True,downscaling_method='skip_pixels',plot=True,plot_type='phase',stop_at_downscaling=None,threshold=1):
    
    return_common_valid_region = remove_null_borders
    
    cumulative_shifts = numpy.zeros((data.shape[0],2))
    
    for downscaling_factor in [int(max_downscaling_factor/(2**i)) for i in range(0,int(numpy.log2(max_downscaling_factor)+1))]: # descreasing list from max_downscaling_factor to 1
        
        neighbor_shifts, total_shift = get_shifts_of_local_variance_parallel(data,fft_upsampling=fft_upsampling,downscaling_factor=downscaling_factor, use_gradient=use_gradient,downscaling_method=downscaling_method,plot=False)

        cumulative_shifts += total_shift
        
        aligned_data_CC = shift_and_crop_volume(data,total_shift,return_common_valid_region=return_common_valid_region, remove_null_borders = remove_null_borders)
        
        if plot:
            angles -= numpy.min(angles)
            
            fig, ax = plt.subplots(2,3,figsize=(20,6),dpi=150)
            ax[0,0].plot(angles[:-1],neighbor_shifts[:-1,0],'.-',label=f'dy')
            ax[0,0].plot(angles[:-1],neighbor_shifts[:-1,1],'.-',label=f'dx')
            ax[0,0].set_title('Neighboor shifts')
            ax[0,0].set_ylabel('pixels * downscaling')
            ax[0,0].grid()
            ax[0,0].legend()
            ax[1,0].plot(angles[:-1],cumulative_shifts[0:-1,0],'-')
            ax[1,0].plot(angles[:-1],cumulative_shifts[0:-1,1],'-')
            ax[1,0].grid()
            ax[1,0].set_title('Cumulative shifts')
            ax[1,0].set_ylabel('pixels * downscaling')
            ax[1,0].set_xlabel('Angle [deg]')
            ax[1,1].set_xlabel('Angle [deg]')
            if plot_type == 'abs':
                ax[0,1].imshow(numpy.abs(data[:,data.shape[1]//2,:]),aspect='auto')
                ax[1,1].imshow(numpy.abs(aligned_data_CC[:,aligned_data_CC.shape[1]//2,:]),aspect='auto')
            elif plot_type == 'phase':
                ax[0,1].imshow(numpy.angle(data[:,data.shape[1]//2,:]),aspect='auto')
                ax[1,1].imshow(numpy.angle(aligned_data_CC[:,aligned_data_CC.shape[1]//2,:]),aspect='auto')
            ax[0,1].set_ylabel('Angle [deg]')
            ax[1,1].set_xlabel('X [pixels]')
            ax[0,1].set_title(f'Downscaling = {downscaling_factor}')
            ax[0,2].imshow(get_VMF_curves(data,use_phase_gradient=use_gradient,filter_sigma=0,curve_portion=None).T,aspect='auto',extent=[angles.min(),angles.max(),0,data.shape[0]])
            ax[1,2].imshow(get_VMF_curves(aligned_data_CC,use_phase_gradient=use_gradient,filter_sigma=0,curve_portion=None).T,aspect='auto',extent=[angles.min(),angles.max(),0,data.shape[0]])
            ax[0,2].set_title('Vertical Mass')
            ax[1,2].set_xlabel('Angle [deg]')
            ax[0,2].set_ylabel('Y [pixels]')
            ax[1,2].set_ylabel('Y [pixels]')            
            plt.tight_layout()
            plt.show()

        data = aligned_data_CC # copy for next iteration
    
        if stop_at_downscaling is not None:
            if stop_at_downscaling == downscaling_factor:
                print('Reached desired downscaling level. Exiting loop...')
                break

        reached_threshold = total_shift < threshold
        if reached_threshold.all() == True:
            print(f'All pixel shifts smaller than {threshold}. Exiting loop.')
            break

    return aligned_data_CC, neighbor_shifts, total_shift

def shift_and_crop_volume(data,total_shift,return_common_valid_region=True, remove_null_borders = True,shift_mode='scipy'):
    if shift_mode == 'scipy':
        aligned_volume = shift_volume_slices_parallel(data,total_shift)
    elif shift_mode == 'fourier':
        total_shift_adjusted = numpy.roll(total_shift,1,axis=0) 
        total_shift_adjusted[0,:] = 0 # first frame suffers no shift.
        aligned_volume = shift_slices_via_fourier_transform_parallel(data,-total_shift_adjusted[:,0],-total_shift_adjusted[:,1],fill_in=0)
    else: 
        raise ValueError("Select a proper shift mode: 'fourier' or 'scipy'")
    
    if return_common_valid_region:
        aligned_volume = find_common_non_null_area(aligned_volume)
    
    if remove_null_borders:
        aligned_volume = remove_zeroed_borders(aligned_volume)

    return aligned_volume



def find_shifts_parallel(frame,next_frame,fft_upsampling=10,use_gradient=False):
    if use_gradient:
        frame = calculate_local_variance_field(frame)
        next_frame = calculate_local_variance_field(next_frame)

    shift, error, diffphase = phase_cross_correlation(frame, next_frame, upsample_factor=fft_upsampling)
    return shift


def get_shifts_of_local_variance_parallel(data,fft_upsampling,downscaling_factor, use_gradient,downscaling_method='skip_pixels',plot=False):
    """ Calculates local variance field of images (in a block) and finds the shift between them.

    Args:
        data (numpy array): block of images to be aligned 
        fft_upsampling (int): upsampling factor to improve sub-pixel imaging registration. See https://doi.org/10.1364/OL.33.000156. 
        downscaling_factor (int, optional): not used for now. Downsamples the image to performe alignment in a lower resolution version of the image. Defaults to 2.

    Returns:
        neighbor_shifts: array of values containing the shift between neighbor images
        total_shifts: shift of each image with respect to the first image
    """
    
    if downscaling_factor > 1:
        print(f"Downscaling images {downscaling_factor} times for alignment...")
        if downscaling_method == 'pyramid_reduce': # slow method! how to speed it up?
            data = pyramid_reduce(numpy.real(data), downscale=downscaling_factor,order=1,channel_axis=0) + 1j*pyramid_reduce(numpy.imag(data), downscale=downscaling_factor,order=1,channel_axis=0)
        elif downscaling_method == 'skip_pixels':
            data = data[:,0::downscaling_factor,0::downscaling_factor]
        else:
            sys.exit('Select a proper downscaling method: pyramid_reduce or skip_pixels')
    
    neighbor_shifts = numpy.zeros((data.shape[0],2))
    
    find_shifts_parallel_partial = partial(find_shifts_parallel,fft_upsampling=fft_upsampling,use_gradient=use_gradient)
        
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(find_shifts_parallel_partial, data[0:-1],data[1::]),total=data.shape[0],desc=f'Finding shifts using {executor._max_workers} workers'))
        for i, shift in enumerate(results):
            neighbor_shifts[i][0] = shift[0]*downscaling_factor
            neighbor_shifts[i][1] = shift[1]*downscaling_factor

        
    total_shift = numpy.cumsum(neighbor_shifts,axis=0)

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].plot(neighbor_shifts[:-1,0],'.-')
        ax[0].plot(neighbor_shifts[:-1,1],'.-')
        ax[1].plot(total_shift[0:-1,0],'o-',label=f'dy')
        ax[1].plot(total_shift[0:-1,1],'o-',label=f'dx')
        ax[1].legend()
        ax[0].grid(), ax[1].grid()
        ax[0].set_title('Neighboor shifts')
        ax[1].set_title('Cumulative shifts')
        ax[0].set_ylabel('Pixels')
        ax[0].set_xlabel('Angle')
        ax[1].set_xlabel('Angle')
    
    return neighbor_shifts, total_shift

def get_shifts_of_local_variance(data,fft_upsampling,downscaling_factor, use_gradient,downscaling_method='skip_pixels',plot=False):
    """ Calculates local variance field of images (in a block) and finds the shift between them.

    Args:
        data (numpy array): block of images to be aligned 
        fft_upsampling (int): upsampling factor to improve sub-pixel imaging registration. See https://doi.org/10.1364/OL.33.000156. 
        downscaling_factor (int, optional): not used for now. Downsamples the image to performe alignment in a lower resolution version of the image. Defaults to 2.

    Returns:
        neighbor_shifts: array of values containing the shift between neighbor images
        total_shifts: shift of each image with respect to the first image
    """
    
    if downscaling_factor > 1:
        print("Downscaling images for alignment...")
        if downscaling_method == 'pyramid_reduce': # slow method! how to speed it up?
            data = pyramid_reduce(numpy.real(data), downscale=downscaling_factor,order=1,channel_axis=0) + 1j*pyramid_reduce(numpy.imag(data), downscale=downscaling_factor,order=1,channel_axis=0)
        elif downscaling_method == 'skip_pixels':
            data = data[:,0::downscaling_factor,0::downscaling_factor]
        else:
            sys.exit('Select a proper downscaling method: pyramid_reduce or skip_pixels')
    
    neighbor_shifts = numpy.zeros((data.shape[0],2))
    
    print('Finding shift between neighboor slices...')
    for i in range(0,data.shape[0]-1):
        if i%50==0: print(f"Finding shift between slices #{i}/{data.shape[0]}")

        if use_gradient:
            local_variance1 = calculate_local_variance_field(data[i])
            local_variance2 = calculate_local_variance_field(data[i+1])
        else:
            local_variance1 = data[i]
            local_variance2 = data[i+1]
        
        shift, error, diffphase = phase_cross_correlation(local_variance1, local_variance2, upsample_factor=fft_upsampling)

        neighbor_shifts[i][0] = shift[0]*downscaling_factor
        neighbor_shifts[i][1] = shift[1]*downscaling_factor
        
    total_shift = numpy.cumsum(neighbor_shifts,axis=0)

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].plot(neighbor_shifts[:-1,0],'.-')
        ax[0].plot(neighbor_shifts[:-1,1],'.-')
        ax[1].plot(total_shift[0:-1,0],'o-',label=f'dy')
        ax[1].plot(total_shift[0:-1,1],'o-',label=f'dx')
        ax[1].legend()
        ax[0].grid(), ax[1].grid()
        ax[0].set_title('Neighboor shifts')
        ax[1].set_title('Cumulative shifts')
        ax[0].set_ylabel('Pixels')
        ax[0].set_xlabel('Angle')
        ax[1].set_xlabel('Angle')
    
    return neighbor_shifts, total_shift

def shift_volume_slices_parallel(data,total_shift):

    # try:
    #     n_cpus = int(os.getenv('SLURM_CPUS_ON_NODE'))
    #     print(f'Using {n_cpus} CPUs')
    # except:
    #     print(f'Could not read CPUs from SLURM. Using {n_cpus} CPUs')

    aligned_volume = numpy.zeros_like(data)
    aligned_volume[0] = data[0]        
        
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(scipy.ndimage.shift,data[1::],total_shift),total=total_shift.shape[0],desc=f'Shifting slices using {executor._max_workers} workers'))

    aligned_volume[1:] = numpy.array(results)

    return aligned_volume

def shift_volume_slices(data,total_shift):
    """ Shifts each image in the block "data" according to the values in total_shift
    """

    aligned_volume = numpy.zeros_like(data)
    aligned_volume[0] = data[0]

    for i in range(0,data.shape[0]-1):
        if i%50==0: print(f"Shifting slice #{i}/{data.shape[0]}")
        aligned_volume[i+1] = scipy.ndimage.shift(data[i+1],total_shift[i])

    return aligned_volume

def calculate_local_variance_field(matrix):
    """ Calculate the local variance field of a complex matrix
    
    """
    
    gradient = numpy.gradient(matrix)
    del_x = gradient[1]
    del_y = gradient[0]
        
    return numpy.sqrt(numpy.abs(del_x)**2 + numpy.abs(del_y)**2)



### VMF: alignment by vertical mass fluctuation ### 

def alignment_vertical_mass_fluctuation(misaligned_volume,angles, filter_sigma = 0, curve_portion = None, use_phase_gradient = False, return_common_valid_region=True, remove_null_borders = True, threshold=1, plot = None):
    """ Performs the alignment via "Vertical Mass Fluctuation" (as presented in https://doi.org/10.1364/OE.27.036637) to refine alignment of images in the vertical direction

    Args:
        misaligned_volume (numpy array): a volume of images, usually already pre-aligned by another method
        use_phase_gradient (bool, optional): whether to use the phase-gradient for alignment instead of the original images. Defaults to False.
        return_common_valid_region (bool, optional): _description_. Defaults to True.
        return_common_valid_region (bool, optional): If true, will return a zeroed border corresponding to the largest shift among all frames in each direction. Defaults to True.
        remove_null_borders (bool, optional): Remove the null borders. Defaults to True.

    Returns:
        aligned_volume (numpy array) : aligned volume
    """
    
    if isinstance(filter_sigma,int):
        filter_sigma_list = [filter_sigma]
    elif isinstance(filter_sigma,list):
        filter_sigma_list = filter_sigma
    else:
        raise ValueError('Please select an integer or list for the filter_sigma input')
    
    total_shift = numpy.zeros((misaligned_volume.shape[0],1))
    
    for filter_sigma in filter_sigma_list:
        print(f'Filtering vertical mass curves with a {filter_sigma}-sigma Gaussian before alignment...')
        curves, aligned_curves, cumulative_shifts, neighbor_shifts = align_1D_curves(misaligned_volume,use_phase_gradient,filter_sigma,curve_portion)

        total_shift += cumulative_shifts

        aligned_volume = numpy.zeros_like(misaligned_volume)
        aligned_volume[0] = misaligned_volume[0]
        for i in tqdm(range(0,misaligned_volume.shape[0]-1),desc='Aligning slices'):
            aligned_volume[i+1] = scipy.ndimage.shift(misaligned_volume[i+1],[cumulative_shifts[i,0],0])

        if return_common_valid_region:
            masked_volume = numpy.where(aligned_volume==0,0,1)
            product = numpy.prod(numpy.abs(masked_volume),axis=0)
            where_null = numpy.where(numpy.abs(product) == 0,0,1)
            aligned_volume[:] = numpy.where(where_null==1,aligned_volume,0) 

        if remove_null_borders:
            aligned_volume = remove_zeroed_borders(aligned_volume)
        
        if plot:

            print('Plotting results...')
            misaligned_curves = get_VMF_curves(misaligned_volume,use_phase_gradient=use_phase_gradient,filter_sigma=filter_sigma,curve_portion=curve_portion)
            aligned_curves = get_VMF_curves(aligned_volume,use_phase_gradient=use_phase_gradient,filter_sigma=filter_sigma,curve_portion=curve_portion)
            angles0 = angles - angles.min()

            fig, ax = plt.subplots(1,4,figsize=(20,6))
            ax[0].plot(angles0[:-1],neighbor_shifts[:-1],'.-')
            ax[0].set_title('Neighboor shifts')
            ax[0].set_ylabel('pixels * downscaling')        
            ax[0].set_xlabel('angle [deg]')
            ax[0].grid()
            ax[1].plot(angles0[:-1],total_shift[:-1],'.-')
            ax[1].grid()
            ax[1].set_title('Cumulative shifts')
            ax[1].set_xlabel('angle [deg]')  
            ax[2].imshow(misaligned_curves.T,vmax=numpy.mean(misaligned_curves)+2*numpy.std(misaligned_curves),vmin=numpy.mean(misaligned_curves)-2*numpy.std(misaligned_curves),aspect='auto',extent=[angles0.min(),angles0.max(),0,misaligned_curves.shape[0]])
            ax[2].set_title('VMF before')
            ax[2].set_ylabel('Y [pixels]')
            ax[2].set_xlabel('angle [deg]')        
            ax[3].imshow(aligned_curves.T,vmax=numpy.mean(aligned_curves)+2*numpy.std(aligned_curves),vmin=numpy.mean(aligned_curves)-2*numpy.std(aligned_curves),aspect='auto',extent=[angles0.min(),angles0.max(),0,aligned_curves.shape[0]])
            ax[3].set_xlabel('angle [deg]')
            ax[3].set_title('VMF after')
            plt.show()
            
        reached_threshold = total_shift < threshold
        if reached_threshold.all() == True:
            print(f'All pixel shifts smaller than {threshold} pixel. Exiting loop.')
            break
            
        misaligned_volume = aligned_volume # copy for next iterations
            
    return aligned_volume, curves, total_shift

def align_1D_curves(misaligned_volume,use_phase_gradient,filter_sigma,curve_portion,fft_upsampling=100):

    curves = get_VMF_curves(misaligned_volume,use_phase_gradient,filter_sigma,curve_portion)

    aligned_curves, total_shift, neighbor_shifts = overlap_curves(curves,fftupsampling=fft_upsampling)

    return curves, aligned_curves, total_shift, neighbor_shifts

def get_VMF_curves(misaligned_volume,use_phase_gradient,filter_sigma,curve_portion):
    curves = []
    for i in range(misaligned_volume.shape[0]):
        frame = misaligned_volume[i]

        if use_phase_gradient:
            curve = vertical_phase_gradient(frame)
            if filter_sigma>0:
                curve = scipy.ndimage.gaussian_filter1d(curve, filter_sigma)
        else:
            curve = vertical_mass_distribution(frame)

        if curve_portion != None:
            if len(curve_portion) != 2:
                sys.exit('Please insert a tuple of 2 values for curve_portion: (value1, value2)')
            elif curve_portion == ():
                pass
            else:
                curve = curve[curve_portion[0]:curve_portion[1]]

        curves.append(curve)

    curves = numpy.asarray(curves)
    return curves

def vertical_phase_gradient(frame):
    """
    Calculate the vertical phase gradient of a complex image using the analytical formula.
    See equation (6) in https://doi.org/10.1364/OE.27.036637 
    """

    gradient = numpy.gradient(frame)
    phase_gradient_y = numpy.imag( frame.conj() * gradient[0] / numpy.abs(frame)**2  )
    phase_gradient_y = numpy.sum(phase_gradient_y,axis=1)
    return phase_gradient_y

def vertical_mass_distribution(frame):
    """
    Calculate vertical mass distrbution of image 
    """
    return numpy.sum(frame,axis=1)
    

def shift_2d_replace(data, dx, dy, constant=False):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: The 2d numpy array to be shifted
    :param dx: The shift in x
    :param dy: The shift in y
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = numpy.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = numpy.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def calculate_curve_ctr_of_mass(curve,positions):
    """
    Calculate the center of mass of a 2D curve
    """
    ctr_mass = numpy.dot(curve,positions)/numpy.sum(curve)
    return ctr_mass


def overlap_curves(data,fftupsampling=100):
    """
    Overlap 2D curves using subpixel image registration methods.
    See https://doi.org/10.1364/OL.33.000156
    """

    def shift_slice(data, shift):
        return scipy.ndimage.shift(data, shift)

    neighbor_shifts = numpy.empty((data.shape[0],1))


    for i in tqdm(range(0,data.shape[0]-1),desc=f'Finding shifts via sub-pixel registration. Upsampling = {fftupsampling}'):
        shift, error, diffphase = phase_cross_correlation(data[i], data[i+1], upsample_factor=fftupsampling,normalization=None) # if normalzaition = phase, shifts are barely > 0
        neighbor_shifts[i][0] = shift[0]

    total_shift = numpy.cumsum(neighbor_shifts,axis=0)
    aligned_curves = numpy.empty_like(data)
    aligned_curves[0] = data[0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a list of futures
        futures = [executor.submit(shift_slice, data[i+1], total_shift[i]) for i in range(data.shape[0] - 1)]
        # Collect results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            aligned_curves[i+1] = future.result()
        
    return aligned_curves, total_shift, neighbor_shifts
