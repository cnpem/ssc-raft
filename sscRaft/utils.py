# General =========
import os
import sys
import numpy
import numpy as np
import json
import h5py
import time
from time import time
import warnings
import pathlib
import inspect
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter,ScalarFormatter

# Multiprocessing =========
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory as SM
import multiprocessing as mp

nthreads = multiprocessing.cpu_count()

# wiggle.py ============
import uuid
import SharedArray as sa
from scipy.optimize import minimize


# alignment.py =========

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

import skimage
import scipy
from skimage.registration import phase_cross_correlation
from skimage.transform import pyramid_gaussian
from skimage.transform import pyramid_reduce
from scipy.ndimage import center_of_mass
# ======================

'''----------------------------------------------'''
import logging

console_log_level = 10

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)
console_handler.setLevel(console_log_level)

DEBUG = 10
'''----------------------------------------------'''

#==========================================================
# Definitions and Constants
#==========================================================
LIGHT_VELOCITY = 299792458                         # [m/s]
PLANK_CONSTANT = 4.135667662E-15                   # [eV.s]
CONST          = PLANK_CONSTANT * LIGHT_VELOCITY   # [eV.m]
#==========================================================

############# Basic functions ##############

def ReconMethod(method):
    if method.lower() == 'none':
        return 0
    elif method.lower() == 'fbp':
        return 1
    elif method.lower() == 'bst':
        return 2
    elif method.lower() == 'eEMRT':
        return 3
    elif method.lower() == 'tEMRT':
        return 4
    elif method.lower() == 'tEMFQ':
        return 5
    elif method.lower() == 'fdk':
        return 6
    else:
        return 1

def RingsMethod(method):
    if method.lower() == 'none':
        return 0
    elif method.lower() == 'all stripes':
        return 1
    elif method.lower() == 'all stripes multiaxis':
        return 2
    elif method.lower() == 'titarenko':
        return 3
    else:
        return 1

def AlignMethod(method):
    if method.lower() == 'none':
        return 0
    elif method.lower() == 'rot axis recon':
        return 1
    elif method.lower() == 'rot axis':
        return 2
    elif method.lower() == 'excentric':
        return 3
    else:
        return 1
    
def BckCorrMethod(method):
    if method.lower() == 'none':
        return 0
    elif method.lower() == 'classic':
        return 1
    elif method.lower() == 'classic no log':
        return 2
    else:
        return 1
    
def FilterNumber(mfilter):
    if mfilter.lower() == 'none':
        return 0
    elif mfilter.lower() == 'gaussian':
        return 1
    elif mfilter.lower() == 'lorentz':
        return 2
    elif mfilter.lower() == 'cosine':
        return 3
    elif mfilter.lower() == 'rectangle':
        return 4
    elif mfilter.lower() == 'hann':
        return 5
    elif mfilter.lower() == 'hamming':
        return 6
    elif mfilter.lower() == 'ramp':
        return 7
    else:
        return 6

def ContrastFilterNumber(mfilter):
    if mfilter.lower() == 'none':
        return 0
    elif mfilter.lower() == 'paganin':
        return 1
    elif mfilter.lower() == 'paganin_slices':
        return 2
    elif mfilter.lower() == 'contrast':
        return 3
    else:
        return 1

def PaddMode(pmode):
    if pmode.lower() == 'none':
        return 0
    elif pmode.lower() == 'zero':
        return 1
    elif pmode.lower() == 'edge':
        return 2
    elif pmode.lower() == 'ones':
        return 3
    else:
        return 2
    
def setInterpolation(name):
    """ Set interpolation 

    Args:
        name (str): string name for the interpolation

    Returns:
        (int): Value defining the interpolation name

    Options:
        name (str): \'nearest\' and \'bilinear\'

    """
    if name.lower() == 'nearest':
        return 0
    elif name.lower() == 'bilinear':
        return 1
    elif name.lower() == 'none':
        return -1
    else:
        logger.warning(f'Interpolation invalid. Using default \'nearest\' interpolation.')
        return 0

def CNICE(darray,dtype=numpy.float32):
    if darray.dtype != dtype:
        return numpy.ascontiguousarray(darray.astype(dtype))
    elif darray.flags['C_CONTIGUOUS'] == False:
        return numpy.ascontiguousarray(darray)
    else:
        return darray

VERBOSE = False
def dprint(*x):
    if VERBOSE:
        print(*x)

def nice(f): # scientific notation + 2 decimals
    return "{:.2e}".format(f)
        
def set_precision(precision):
    """Select datatype of numpy array.

    Args:
        precision (str): precision name

    Returns:
        (int): Integer value defining the numpy datatype
        (type): The numpy datatype

    Raises:
        Invalid datatype

    Options:
        precision (str): \'float32\', \'uint16\', \'uint8\'
    """
    if precision == 'float32':
        datatype = 5
        Ndatatype = numpy.float32
    elif precision == 'uint16':
        datatype = 2
        Ndatatype = numpy.uint16
    elif precision == 'uint8':
        datatype = 1
        Ndatatype = numpy.uint8
    else:
        logger.error(f'Invalid datatype:{precision}. Options: `float32`, `uint16` and `uint8`.')
        raise ValueError(f'Invalid datatype:{precision}. Options: `float32`, `uint16` and `uint8`.')

    return datatype, Ndatatype

def sanitize_angles(angles: numpy.ndarray) -> numpy.ndarray:
    ''' Set angles to start at 0 (zero) radians. 
    '''
    angles = angles - min(angles)

    return angles

def get_excentric_obj_dimensions(nx,ny):
     return 2 * nx, 2 * ny

def get_padding_size(n,pad):
     return int( (pad * n) / 100 )

def set_input_axis_order(order: str = 'slices_angles_length'):
    # Default is 'slices_angles_length'

    if order.lower() == 'angles_slices_length':
        return 0
    elif order.lower() == 'slices_angles_length':
        return 1
    elif order.lower() == 'transpose_length':
        return 2
    else:
        message = f'The axis input parameter \'{order}\' does not exist. Setting \'slices_angles_length\' as default.'
        logger.warning(message)
        return 1 
    
def power_of_2_padding(size,pad):
    return int((pow(2, numpy.ceil(numpy.log2(size + 2 * pad))) - size) * 0.5)
       
def crop_circle(data, z1, zt, pixeldet, npixels = 0):
    r"""Function to crop a circle of radius :math:`r` on the xy-plane of a 3D data.
        
        - r = (radius - npixels * pixel_size);
        - radius = (pixel_size * dimension) / 2;
        - pixel_size = pixel_detector * magnitude;
        - magnitude = z1/zt
        - zt = z1+z2

    Args:
        data (ndarray): 3D data
        z1 (float): Sample-source distance in meters
        zt (float): Detector-source distance in meters (z1+z2)
        pixeldet (float): Detector pixel size in meters 
        npixels (int, optional): How many pixels to subtract from radius to reduce cropped circle [Default: 0]

    Returns:
        (ndarray): 3D cropped data
    """
    dimx = data.shape[-1]
    dimy = data.shape[-2]

    pixel_size = pixeldet * ( z1 / zt )

    diameterx = dimx * pixel_size
    radiusx = diameterx / 2.0

    diametery =  dimy * pixel_size
    radiusy = diametery / 2.0

    x = numpy.linspace(-radiusx,radiusx,dimx)
    y = numpy.linspace(-radiusy,radiusy,dimy)

    Xm, Ym = numpy.meshgrid(x,y)
    radius = max(radiusx,radiusy)
    r = radius - npixels * pixel_size

    mask = (Xm)**2 + (Ym)**2  <= r**2

    for i in range(data.shape[0]):
            data[i,:,:] = numpy.where(mask,data[i,:,:],0)

    return data

def crop_ellipse(data, pixeldet, magn = 1, npixelsx = 0, npixelsy = 0):
    r"""Function to crop an ellipse of radius :math:`a` and :math:`b` on the xy-plane of a 3D data.

    Args:
        data (ndarray): 3D data
        pixeldet (float tuple): Detector pixel size (x,y) in x- and y- directions in meters 
        magn (float): Magnification related to beam geometry (z1+z2)/z1 [Default: 1, parallel geometry]
        npixelsx (int, optional): How many pixels to subtract from radius ``a`` (x-direction) to reduce cropped circle [Default: 0]
        npixelsy (int, optional): How many pixels to subtract from radius ``b`` (y-direction) to reduce cropped circle [Default: 0]

    Returns:
        (ndarray): 3D cropped data
    """
    dimx = data.shape[-1]
    dimy = data.shape[-2]

    pixel_sizex = pixeldet[0] * ( 1 / magn )
    pixel_sizey = pixeldet[1] * ( 1 / magn )

    diameterx = dimx * pixel_sizex
    a = diameterx / 2.0 - npixelsy * pixel_sizey

    diametery =  dimy * pixel_sizey
    b = diametery / 2.0 - npixelsx * pixel_sizex

    x = numpy.linspace(-a,a,dimx)
    y = numpy.linspace(-b,b,dimy)

    Xm, Ym = numpy.meshgrid(x,y)

    mask = (Xm/a)**2 + (Ym/b)**2  <= 1

    for i in range(data.shape[0]):
            data[i,:,:] = numpy.where(mask,data[i,:,:],0)
            
    return data

def crop_circle_simple(data, pixel_size, npixels = 0):
    r"""Function to crop a circle of radius :math:`r` on the xy-plane of a 3D data.
        
        - r = (radius - npixels * pixel_size);


    Args:
        data (ndarray): 3D data
        pixel_size (float): Pixel size in meters 
        npixels (int, optional): How many pixels to subtract from radius to reduce cropped circle [Default: 0]

    Returns:
        (ndarray): 3D cropped data
    """
    dimx = data.shape[-1]
    dimy = data.shape[-2]

    diameterx = dimx * pixel_size
    radiusx = diameterx / 2.0

    diametery =  dimy * pixel_size
    radiusy = diametery / 2.0

    x = numpy.linspace(-radiusx,radiusx,dimx)
    y = numpy.linspace(-radiusy,radiusy,dimy)

    Xm, Ym = numpy.meshgrid(x,y)
    radius = max(radiusx,radiusy)
    r = radius - npixels * pixel_size

    mask = (Xm)**2 + (Ym)**2  <= r**2

    for i in range(data.shape[0]):
            data[i,:,:] = numpy.where(mask,data[i,:,:],0)

    return data

def plot_imgs(images, titles = None, limits = None, save_name = None, pixel = None, maxcols = 4, cmap = 'viridis'):
    """Function for multiple plots.

    Args:
        images (list): List of images. Ex: \'[img1,img2]\'
        titles (list of str,optional): List of image titles. Ex: \'['title img1','title img2']\'. [default: None]
        limits (list,optional): Listo of data range. Ex:  \'[[0,1],[-2,2]]\'. Can be used as \'[[0,1],None]\' [default: None]
        save_name (str,optional): Filename of figure for saving in png [default: None]
        pixel (str,optional): Pixel size in meters for scalebar [default: None]
        maxcols (int,optional): Plot will be displayed in a grid form with maxcols columns [default: 4]
        cmap (str,optional): Colormap instance, same as the matplotlib package [default: \'viridis\']

    Returns:
        None

    """
    n     = len(images)
    ncols = min(n, maxcols)
    nrows = int(numpy.ceil(n / ncols))

    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 8))
    # ax      = ax.flatten() # Flatten the 2D array of axes for easier iteration

    if(n == 1):
        ax = [ax]
    else:
        ax = ax.flatten() # Flatten the 2D array of axes for easier iteration

    for i in range(n):
        if limits is not None:
            if limits[i] is not None:
                im = ax[i].imshow(images[i], vmin = limits[i][0], vmax = limits[i][1], cmap = cmap)
            else:
                im = ax[i].imshow(images[i], cmap = cmap) 
        else:
            im = ax[i].imshow(images[i], cmap = cmap)
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax[i].set_xlim(0,images[i].shape[1])
        ax[i].set_ylim(images[i].shape[0],0)

        if pixel is not None:
            scalebar = ScaleBar(pixel, units="m", dimension="si-length", box_alpha=0.5, location="lower left", length_fraction=0.25) # 1 pixel = 0.2 meter
            ax[i].add_artist(scalebar)
        
        if titles != None:
            ax[i].set_title(titles[i])

    for j in range(n, len(ax)):
        fig.delaxes(ax[j])
        
    if(save_name):
        plt.savefig(save_name +'.png', bbox_inches='tight', pad_inches=0, dpi=300)
    
    plt.show()


def lineplot(images, labels, line_number, title = None, save_name = None, reference = None, ref_label = None):
    """Function for multiple lineplots in one figure.

    Args:
        images (list,optional): List of images. Ex: \'[img1,img2]\'
        labels (list of str): List of image labels. Ex: \'['label img1','label img2']\'
        line_number (int): Line number for the plots 
        title (str,optional): Image title [default: None]
        save_name (str,optional): Filename of figure for saving in png [default: None]
        reference (ndarray,optional): Reference image [default: None]
        ref_label (str,optional): Reference image label [default: None]

    Returns:
        None

    """
    # Define custom colors
    colors    = ['#3498db', '#ff796c', '#27ae60', '#c79fef', '#8c000f']
    linestyle = ['solid', 'dashed', 'dotted', 'dashdot']
    
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font

    ndata = len(images)
    
    # Create a figure and axis
    plt.figure(figsize=(20, 5))

    ny, nx = images[0].shape
    
    x = np.linspace(0,nx-1, nx)

    for i in range(0, ndata):
            
        line = images[i][line_number,:] 
        # plt.plot(x, line, label=label[i], linewidth=1.5, linestyle=linestyle[i], alpha = 0.99)
        plt.plot(x, line, label=labels[i], linewidth=1.5, alpha = 0.99)

    if reference is not None:
        line = reference[line_number,:] 
        if ref_label is not None:
            # plt.plot(x, line, label=label[i], linewidth=1.5, linestyle=linestyle[0], alpha = 0.99)
            plt.plot(x, line, label=ref_label, linewidth=1.5, alpha = 0.99)
        else:
            # plt.plot(x, line, label=label[i], linewidth=1.5, linestyle=linestyle[0], alpha = 0.99)
            plt.plot(x, line, label='Reference', linewidth=1.5, alpha = 0.99)

    # # Add labels and title
    plt.xlabel('Pixel', fontsize=15)
    plt.ylabel('Value', fontsize=15)
    if title is not None:
        plt.title(title, fontsize=18)

    # Customize the legend
    plt.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
    # plt.legend('')

    # Add gridlines with customization
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add spines for aesthetics
    ax = plt.gca()
    ax.spines['top'].set_visible('#34495e')
    ax.spines['right'].set_visible('#34495e')
    ax.spines['left'].set_color('#34495e')
    ax.spines['bottom'].set_color('#34495e')
    # ax.axis('off')
    # Hide the tick labels on both axes
    plt.gca().tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    plt.gca().tick_params(axis='y', which='both', left=True, right=False, labelleft=True)

    # Adjust tick parameters
    plt.tick_params(axis='both', which='major', labelsize=12, color='#34495e')

    if(save_name):
            plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
            
    # Show the plot
    plt.show()


def histogram(images, titles, save_name = None, bins = 200, maxcols = 2):
    """Function for multiple histograms.

    Args:
        images (list): List of images. Ex: \'[img1,img2]\'
        titles (list of str,optional): List of image titles. Ex: \'['title img1','title img2']\'. [default: None]
        save_name (str,optional): Filename of figure for saving in png [default: None]
        bins (int,optional): Number of bins [default: 200]
        maxcols (int,optional): Plot will be displayed in a grid form with maxcols columns [default: 2]

    Returns:
        None

    """
    n     = len(images)
    ncols = min(n, maxcols)
    nrows = int(numpy.ceil(n / ncols))

    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 20, nrows * 5))

    if(n == 1):
        ax = [ax]
    else:
        ax = ax.flatten() # Flatten the 2D array of axes for easier iteration

    # Define custom colors
    colors = ['#3498db', '#ff796c', '#27ae60', '#c79fef', '#8c000f']

    for i in range(n):
        # Plot histogram for the first dataset with improved aesthetics
        im = ax[i].hist(images[i].flatten(), bins=bins, alpha=0.9, color=colors[2], edgecolor='white',linewidth=2.2)

        # Customize the y-axis to display a multiplication factor
        ax[i] = plt.gca()
        ax[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax[i].yaxis.get_offset_text().set_fontsize(12)  # Set the font size of the multiplier label
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # ax.axis('off')

        # # Add labels and title with enhanced font sizes
        ax[i].set_xlabel(r'$\text{Value}$', fontsize=16)
        ax[i].set_ylabel(r'$\text{Frequency}$', fontsize=16)
        ax[i].set_title(titles[i], fontsize=18)

    # Hide the tick labels on both axes
    plt.gca().tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    plt.gca().tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    
    # Customize the legend
    # plt.legend(fontsize=12, loc='upper left')

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)

    if(save_name):
        plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # Show the plot
    plt.show()

def squarify(data):
    
    dim_size = len(data.shape)
    
    if dim_size == 3:
        nz, ny, nx = data.shape
        print(f'Data shape: (nz,ny,nx) = ({nz},{ny},{nx})')

        # Compute largest dimension
        largest_dim = max( nz, max( ny, nx ) )

        print(f'Largest dimension is: {largest_dim}')

        # compute padding
        # Padx
        padx = (largest_dim - nx) // 2
        # Pady
        pady = (largest_dim - ny) // 2
        # Padz
        padz = (largest_dim - nz) // 2

        print(f'Padding value: (padz,pady,padx) = ({padz},{pady},{padx})')

        # Apply padding in the volume
        padding = ((padz,padz),(pady,pady),(padx,padx)) # Dimensions for padd: ((padz_left,padz_right),(pady_left,pady_right),(padx_left,xpad_right))
        
        padded_data = np.pad(data, padding, mode='edge')      # Padding coping the last column/row
        
        pnz, pny, pnx = padded_data.shape
        print(f'Padded data new shape: (nz,ny,nx) = ({pnz},{pny},{pnx})')
        
    elif dim_size == 2:
        ny, nx = data.shape
        print(f'Data shape: (ny,nx) = ({ny},{nx})')

        # Compute largest dimension
        largest_dim = max( ny, nx )

        print(f'Largest dimension is: {largest_dim}')

        # compute padding
        # Padx
        padx = (largest_dim - nx) // 2
        # Pady
        pady = (largest_dim - ny) // 2

        print(f'Padding value: (pady,padx) = ({pady},{padx})')

        # Apply padding in the volume
        padding = ((pady,pady),(padx,padx)) # Dimensions for padd: ((pady_left,pady_right),(padx_left,xpad_right)) 
        
        padded_data = np.pad(data, padding, mode='edge')      # Padding coping the last column/row
        
        pny, pnx = padded_data.shape
        print(f'Padded data new shape: (ny,nx) = ({pny},{pnx})')
        
    else:
        print(f'Input data must be 3- or 2-dimensional. Data shape is {data.shape}')
        print('Returning original data')
        
        padded_data = data
    
    return padded_data