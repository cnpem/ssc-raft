from ...rafttypes import *

from ...geometries.parallel import fbp_methods, radon_methods

try:
    import tomopy
except:
    import warnings
    warnings.warn("Could not import Tomopy. Do not try to use tomopy for Iterative Reprojection alignment.")

from .common import find_common_non_null_area, remove_zeroed_borders, shift_slices_via_fourier_transform_parallel

def derivative_via_fourier_transform(data):
    ft = numpy.fft.fft2(data)
    freqy = numpy.fft.fftfreq(data.shape[0])
    freqx = numpy.fft.fftfreq(data.shape[1])
    XX, YY = numpy.meshgrid(freqx,freqy)
    grad_x = numpy.fft.ifft2(2*numpy.pi*XX*ft)
    grad_y = numpy.fft.ifft2(2*numpy.pi*YY*ft)
    return (numpy.real(grad_y), numpy.real(grad_x))

def derivative_fft(arr, axis=0):

    plt.figure()
    plt.imshow(arr)
    
    # Perform 2D Fourier transform
    arr_fft = numpy.fft.fft2(arr)

    # Create frequency grid
    nx, ny = arr.shape
    kx = numpy.fft.fftfreq(nx).reshape(-1, 1)
    ky = numpy.fft.fftfreq(ny).reshape(1, -1)

    # Apply derivative in the frequency domain
    if axis == 0:
        arr_fft *= 1j * 2 * numpy.pi * ky
    elif axis == 1:
        arr_fft *= 1j * 2 * numpy.pi * kx
    else:
        raise ValueError("Axis must be 0 or 1")

    # Perform inverse Fourier transform
    derivative = numpy.real(numpy.fft.ifft2(arr_fft))

    return derivative

def high_pass_filter(img,cutoff=10):
    if 1:
        return img
    else:
        mask=numpy.zeros(img.shape)
        y,x = numpy.linspace(0,img.shape[0]-1,img.shape[0])-img.shape[0]//2,numpy.linspace(0,img.shape[1]-1,img.shape[1])-img.shape[1]//2
        Y,X = numpy.meshgrid(y,x)
        mask = numpy.where(X**2+Y**2<cutoff**2,0,1)
        FT = numpy.fft.fftshift(numpy.fft.fft2(img))
        filtered = mask*FT
        IFT = numpy.fft.ifft2(numpy.fft.ifftshift(filtered))
        return numpy.real(IFT)

def get_displacement(data,reprojected_data,weight_matrix):
    # grad_y, grad_x = derivative_via_fourier_transform(reprojected_data)
    # grad_y, grad_x = derivative_fft(reprojected_data,axis=0), derivative_fft(data,axis=1)
    grad_y, grad_x = numpy.gradient(reprojected_data)
    
    dx = numpy.sum(weight_matrix**2*high_pass_filter(grad_x)*high_pass_filter(data-reprojected_data))/numpy.sum(weight_matrix**2*(high_pass_filter(data-reprojected_data))**2)
    dy = numpy.sum(weight_matrix**2*high_pass_filter(grad_y)*high_pass_filter(data-reprojected_data))/numpy.sum(weight_matrix**2*(high_pass_filter(data-reprojected_data))**2)  
    
    return dy, dx

def find_shift(data,reprojected_data, cumulative_shifts, downsampling,method='correlation_parallel',fft_upsampling=100):
    
    shifts = numpy.empty_like(cumulative_shifts)
    
    downsampled_data = data[:,0::downsampling,0::downsampling]
    downsampled_reprojected_data = reprojected_data[:,0::downsampling,0::downsampling]
    
    if method == 'correlation':
        for i, slices in enumerate(downsampled_data):
            
            slice1, slice2 = downsampled_data[i], downsampled_reprojected_data[i]

            shift, error, _ = skimage.registration.phase_cross_correlation(slice2,slice1,upsample_factor=fft_upsampling,normalization=None)
            
            shifts[i,0:2] = shift*downsampling
            cumulative_shifts[i,0:2] += shifts[i,0:2]
            
    elif method == 'correlation_parallel':
        
        phase_cross_correlation_partial = partial(skimage.registration.phase_cross_correlation,upsample_factor=fft_upsampling,normalization=None)
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(phase_cross_correlation_partial, downsampled_reprojected_data,downsampled_data),total=downsampled_data.shape[0],desc='Search in progress...'))
            for i, result in enumerate(results):
                shifts[i, 0:2] = result[0]*downsampling
                cumulative_shifts[i,0:2] += shifts[i,0:2]

    elif method == 'minimization':
        # See DOI: 10.1364/OE.27.036637
        for i, slices in enumerate(zip(downsampled_reprojected_data,downsampled_data)):
            weight_matrix = numpy.ones_like(downsampled_data) # TODO
            slice1, slice2 = slices
            shift = get_displacement(slice1,slice2,weight_matrix)
            # print(shift)
            cumulative_shifts[i,0], cumulative_shifts[i,1] = shift[0]*downsampling, shift[1]*downsampling
    else:
        sys.exit('Select proper method for finding shift')
     
    return cumulative_shifts, shifts

def call_shift(args):
    array, shift_y, shift_x = args
    return scipy.ndimage.shift(array, (shift_y, shift_x))

def shift_slices_via_fourier_transform(data,shift_y, shift_x):
    ft_data = numpy.fft.fftshift(numpy.fft.fft2(data),axes=(1,2))
    ft_freq_y = numpy.fft.fftshift(numpy.fft.fftfreq(data.shape[1],d=1))
    ft_freq_x = numpy.fft.fftshift(numpy.fft.fftfreq(data.shape[2],d=1))
    XX, YY = numpy.meshgrid(ft_freq_x,ft_freq_y)
    
    exponential_part = numpy.exp(1j * 2 * numpy.pi * (shift_y[:, numpy.newaxis, numpy.newaxis] * YY + shift_x[:, numpy.newaxis, numpy.newaxis] * XX))
    shifted_data = numpy.fft.ifft2(numpy.fft.ifftshift(ft_data*exponential_part))

    return numpy.abs(shifted_data)


def apply_shifts(data,shifts, method='scipy',cpus=32,turn_off_vertical=False,remove_null_borders=True):
    
    if turn_off_vertical:
        shifts[:,0] = 0

    if method == 'scipy':
        for i in range(0,shifts.shape[0]):
            data[i] = scipy.ndimage.shift(data[i],(shifts[i,0],shifts[i,1]))
    elif method == 'scipy_parallel':
    
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            results = list(tqdm(executor.map(call_shift, zip(data,shifts[:,0],shifts[:,1])),total=data.shape[0],desc='Shifting in progress...'))
            for counter, result in enumerate(results):
                data[counter, :, :] = result
    elif method == 'fourier_parallel':
        data = shift_slices_via_fourier_transform(data,-shifts[:,0],-shifts[:,1]) # not filling in values with 0
    elif method == 'fourier_parallel':
        data = shift_slices_via_fourier_transform_parallel(data,-shifts[:,0],-shifts[:,1],fill_in = 0)
    else:
        sys.exit('Select alignment method correctly')

    if remove_null_borders:
        data = find_common_non_null_area(data)
        data = remove_zeroed_borders(data)

    return data

def reproject(tomogram, angles,radon_method='raft',cpus=1,gpus=[0]):
    print('Reprojecting...')
    if radon_method == 'raft':
        tomogram = radon_methods(tomogram, angles, gpus)
        tomogram = numpy.swapaxes(tomogram,0,1)
    elif radon_method == 'tomopy':
        tomogram = tomopy.sim.project.project(tomogram,angles,ncore=cpus,pad=False)
    else:
        raise ValueError('Select a proper method for projection/radon transform: raft or tomopy')
    return tomogram

def reconstruct_and_reproject(data,angles,dic,tomo_method='raft',radon_method='raft',cpus=1,gpus=[0]):
    print('Reconstructing...')
    if tomo_method == 'raft':
        tomo = fbp_method(numpy.swapaxes(data,0,1), dic["algorithm_dic"])
        # tomo = sscRaft.bst(numpy.swapaxes(data,0,1), dic["algorithm_dic"])
    elif tomo_method == 'tomopy':
        tomo = tomopy.recon(data,angles,algorithm='fbp',ncore=cpus,filter_name='ramlak')
    else:
        raise ValueError('Select a proper method for tomographic reconstruction: raft or tomopy')
    reprojected_data = reproject(tomo,angles,radon_method,cpus,gpus)
    return tomo, reprojected_data

def remove_black_borders(volume):
    """
    Remove the null borders of a volume of images 
    """

    not_null = numpy.argwhere(numpy.abs(volume[0]))

    # Bounding box of non-black pixels.
    x0, y0 = not_null.min(axis=0)
    x1, y1 = not_null.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    volume = volume[:,x0:x1, y0:y1]

    return volume

def check_sinogram_shape(sinogram):
    
    myshape = sinogram.shape

    if myshape[1]%2 != 0 or myshape[2]%2 != 0:

        if myshape[1]%2 != 0:
            sinogram = sinogram[:,0:-1,:]
        if myshape[2]%2 != 0:
            sinogram = sinogram[:,:,0:-1]

        print(f"As of now, filtering in sscRaft requires an array with even pixels. Your array shape is {myshape}. Adjusting shape to {sinogram.shape} \n")
            
    return sinogram

def gradient_y_wrapping_insensitive(sinogram,pixel_size):
    g = numpy.exp(1j*sinogram) 
    return numpy.angle(g * numpy.conj(numpy.roll(g,shift=1,axis=-1)))/pixel_size

def phase_derivative_hilbert_transform(sinogram,pixel_size=1):
    """
    Calculate phase derivative to use it with backprojection (no filter!) 
    Pixel size in meters
    """
    grad_sinogram = gradient_y_wrapping_insensitive(sinogram,pixel_size)
    hilbert = numpy.imag(scipy.signal.hilbert(grad_sinogram,axis=-1)/(2*numpy.pi))
    return hilbert


def plot_iterative_reprojection(tomo,sinogram,angles,neighboor_shifts,cumulative_shifts,plot_type=None):
    
    angles = angles*180/numpy.pi
    
    # if plot_type is None:
    #     pass
    # elif plot_type == 'phase':
    #     sinogram = numpy.angle(sinogram)
    # elif plot_type == 'amplitude':
    #     sinogram = numpy.abs(sinogram)
    # else:
    #     raise ValueError('Select correct plot_type')

    fig, ax = plt.subplots(2,4,figsize=(12,6),dpi=150)
    
    ax[0,0].plot(angles,neighboor_shifts[:,0],'.', label=f'y RMS={neighboor_shifts[:,0].std():.2e}')
    ax[0,0].plot(angles,neighboor_shifts[:,1],'.', label=f'x RMS={neighboor_shifts[:,1].std():.2e}')

    ax[0,1].plot(angles,cumulative_shifts[:,0],'.', label='dy')
    ax[0,1].plot(angles,cumulative_shifts[:,1],'.', label='dx')
    
    ax[0,2].imshow(sinogram[:,sinogram.shape[1]//2,:],aspect='auto',extent=[angles[0],angles[-1],0,sinogram.shape[0]])
    ax[0,3].imshow(tomo[tomo.shape[0]//2,:,:],aspect='auto')
    
    ax[1,0].imshow(tomo.sum(0))
    ax[1,1].imshow(tomo.sum(1))    
    ax[1,2].imshow(tomo.sum(2))
    
    ax[0,0].grid()
    ax[0,0].set_ylabel('shift [pxls]')
    ax[0,0].set_title('Neighboor shifts')
    ax[0,0].set_xlabel('angle [deg]')
    ax[0,0].legend()
    # ax[0,0].set_aspect()

    ax[0,1].set_xlabel('angle [deg]')
    ax[0,1].set_title('Cumulative shifts')
    ax[0,1].grid()
    # ax[0,1].set_aspect()
    
    ax[0,2].set_title('Sinogram')
    ax[0,2].set_ylabel('angles [deg]')
    ax[0,2].set_xlabel('X [pixels]')

    ax[0,3].set_title('Tomo central Z slice')
    ax[0,3].set_ylabel('Y [pixels]')
    ax[0,3].set_xlabel('X [pixels]')
    
    ax[1,0].set_title('Sum Z')
    ax[1,0].set_ylabel('Y')
    ax[1,0].set_xlabel('X') 
    
    ax[1,1].set_title('Sum Y')    
    ax[1,1].set_ylabel('Z')
    ax[1,1].set_xlabel('X') 

    ax[1,2].set_title('Sum X')   
    ax[1,2].set_ylabel('Z')
    ax[1,2].set_xlabel('Y') 
    
    ax[1,3].axis('off')
    
    plt.tight_layout()
    plt.show()

def iterative_reprojection(original_sinogram,angles,gpus=[0],n_cpus=32, using_phase_derivative=False,threshold=1e-2, max_iterations=10, max_downsampling=1,fft_upsampling=100,turn_off_vertical=False,plot=False,plot_type='phase', FBP_filter='lorentz',find_shift_method='correlation',apply_shift_method='scipy',tomo_method='raft',radon_method='raft'):

    try:
        import tomopy
    except:
        raise ValueError('"Iterative Reprojection may need tomopy. Please install it first')

    try:
        n_cpus = int(os.getenv('SLURM_CPUS_ON_NODE'))
        print(f'Using {n_cpus} CPUs')
    except:
        print(f'Could not read CPUs from SLURM. Using {n_cpus} CPUs')

    dic = {}
    dic["algorithm_dic"] = {}
    dic["algorithm_dic"]['algorithm'] = "FBP"
    if 'regularization' not in dic["algorithm_dic"]:
        dic["algorithm_dic"]['paganin regularization'] = 0 # regularization <= 1; use for smoothening
    if using_phase_derivative==False:
        dic["algorithm_dic"]['filter'] = FBP_filter
    else:
        dic["algorithm_dic"]['filter'] = "" # no filtering, i.e., doing backprojection only

    dic["algorithm_dic"]['gpu'] =  gpus
    dic["algorithm_dic"]['angles'] =  angles # angles in radians
    # dic["algorithm_dic"]['angles'] -= dic["algorithm_dic"]['angles'].min()
    
    fig, ax0 = plt.subplots()
    ax0.plot(dic["algorithm_dic"]['angles'], 'o-')
    ax0.set_title('Make sure your angles are in radians')
    ax0.axhline(numpy.pi/2,label=r'$90^\circ = \pi/2$',color='red')
    ax0.axhline(-numpy.pi/2,label=r'$-90^\circ = -\pi/2$',color='red')
    # ax0.axhline(0,label='0',color= 'red')
    ax0.grid()
    ax0.legend()
    ax0.set_ylabel('Angle value')
    ax0.set_xlabel('Frame number')
    plt.show()
    
    original_sinogram = check_sinogram_shape(original_sinogram.copy())
    sinogram = original_sinogram.copy()

    if using_phase_derivative:
        print('Computing phase derivative of sinogram...')
        sinogram=phase_derivative_hilbert_transform(sinogram,pixel_size=1)

    print('Reconstructing and reprojecting from input data...')
    tomo, reprojected_sinogram = reconstruct_and_reproject(sinogram,dic["algorithm_dic"]["angles"],dic,tomo_method=tomo_method,radon_method=radon_method,cpus=n_cpus,gpus=dic['algorithm_dic']['gpu'])

    cumulative_shifts = numpy.empty((original_sinogram.shape[0],2),dtype=numpy.float32)
    cumulative_shifts[:,0] = 0  
    cumulative_shifts[:,1] = 0  
    
    plot_iterative_reprojection(tomo,original_sinogram,dic["algorithm_dic"]['angles'],cumulative_shifts,cumulative_shifts,plot_type=plot_type) # plot initial recon

    
    iter_count = 0 
    for downsampling in [int(max_downsampling/(2**i)) for i in range(0,int(numpy.log2(max_downsampling)+1))]: # descreasing list from max_downsampling to 1
        print(f'Downsampling data {downsampling} times...')
        while iter_count < max_iterations:
            print(f'Iteration #',iter_count+1)

            print('Finding shifts...')
            
            if using_phase_derivative:
                # sinogram_adjusted = adjust_data_for_correlation(sinogram,using_phase_gradient=using_phase_derivative)
                # reprojected_adjusted = adjust_data_for_correlation(reprojected_sinogram,using_phase_gradient=using_phase_derivative)
                # cumulative_shifts,shifts = find_shift(sinogram_adjusted, reprojected_adjusted, cumulative_shifts, downsampling,method=find_shift_method,fft_upsampling=fft_upsampling)
                cumulative_shifts,shifts = find_shift(sinogram, reprojected_sinogram, cumulative_shifts, downsampling,method=find_shift_method,fft_upsampling=fft_upsampling)
            else:
                cumulative_shifts,shifts = find_shift(sinogram, reprojected_sinogram, cumulative_shifts, downsampling,method=find_shift_method,fft_upsampling=fft_upsampling)

                
            reached_threshold = shifts < threshold
            if reached_threshold.all() == True:
                print(f'All pixel shifts smaller than {threshold}. Exiting algorithm.')
                break # exit while loop
            
            print('Applying shifts...')
            sinogram = apply_shifts(original_sinogram.copy(),cumulative_shifts,method=apply_shift_method,turn_off_vertical=turn_off_vertical)
            sinogram = check_sinogram_shape(sinogram)

            print('Reconstructing and reprojecting...')
            if using_phase_derivative:
                phase_derivative_sinogram=phase_derivative_hilbert_transform(sinogram,pixel_size=1)
                tomo, reprojected_sinogram = reconstruct_and_reproject(phase_derivative_sinogram,dic["algorithm_dic"]["angles"],dic,tomo_method=tomo_method,radon_method=radon_method)
            else:
                tomo, reprojected_sinogram = reconstruct_and_reproject(sinogram,dic["algorithm_dic"]["angles"],dic,tomo_method=tomo_method,radon_method=radon_method)
                
            fig, ax = plt.subplots(1,3,figsize=(12,4),dpi=200)
            im0 = ax[0].imshow(sinogram[0])
            fig.colorbar(im0,ax=ax[0])
            im1=ax[1].imshow(reprojected_sinogram[0])
            fig.colorbar(im1,ax=ax[1])            
            im2=ax[2].imshow(numpy.angle(reprojected_sinogram[0]))
            fig.colorbar(im2,ax=ax[2])
            
            if plot:
                plot_iterative_reprojection(tomo,sinogram,dic["algorithm_dic"]['angles'],shifts,cumulative_shifts,plot_type=plot_type)
                
            iter_count += 1
            
        if reached_threshold.all() == True:
            break # exit for loop

        aligned_tomo = fbp_method(numpy.swapaxes(sinogram,0,1), dic["algorithm_dic"])
        
    return aligned_tomo, sinogram, cumulative_shifts, reprojected_sinogram
