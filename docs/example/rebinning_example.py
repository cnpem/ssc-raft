import numpy as np
from sscRaft import rebinning as rb
from sscRadon import *

def phantom_ball(size):
    phantom = np.zeros((size,size,size))
    xx = np.linspace(-1,1,size)
    yy = np.linspace(-1,1,size)
    zz = np.linspace(-1,1,size)

    Xm, Ym, Zm = np.meshgrid(xx,yy,zz)
    mask = (Xm)**2 + (Ym)**2 + (Zm)**2 <= (0.8)**2 
    phantom = np.where(mask,4,0)
    mask = (Xm-0.2)**2 + (Ym-0.2)**2 + (Zm)**2 <= (0.4)**2 
    phantom += np.where(mask,1,0)
    mask = (Xm+0.3)**2 + (Ym+0.3)**2 + (Zm)**2 <= (0.3)**2 
    phantom += np.where(mask,2,0)

    return phantom

# ======== Create Phantom ==================================================
size =  64
phantom = phantom_ball(size)
# phantom  = mario.createMario(shape=size, noise=False, zoom=0.3)
# phantom = phantom_square(size,10)

# ======== General Conical Tomogram Parameters Dictionary ===========================================
# ========     Works for both ConeRadon and Rebinning     ===========================================
dic = {}  # Declare Dictionary

angle = 0 # (Coneradon) Angle in Degrees of the Cone Projection (2D) you want to obtain
ttype = 'conical' # (Coneradon) Chose type of Radon transform: 'conical' for coneradon, 'parallel' for radon

dic['Distances'] = (2,1) # (Coneradon/Rebinning) (z1, z2) Distances source/sample (z1) and sample/detector (z2) 
dic['Poni'] = (0.5,0) # (Coneradon/Rebinning) Tuple PONI (point of incidence) of central ray at detector (cx,cy)
dic['ShiftPhantom'] = (0.2,0) # (Coneradon/Rebinning) Tuple of phantom shift (sx,sy)
dic['ShiftRotation'] = (1,0)# (Coneradon/Rebinning) Tuple of rotation center shift (rx,ry)
dic['DetectorSize'] = (1,1) # (Coneradon/Rebinning) Tuple of detector size (Dx,Dy), where the size interval is [-Dx,Dx], [-Dy,Dy]
dic['ParDectSize'] = dic['DetectorSize'] # (Coneradon/Rebinning) Tuple of detector size (Lx,Ly), where the size interval is [-Lx,Lx], [-Ly,Ly]
dic['PhantomSize']  = (1,1,1) # (Coneradon/Rebinning) Tuple of phantom size (Lx,Ly,Lz), where the size interval is [-Lx,Lx], [-Ly,Ly], [-Lz,Lz]

dic['Type'] = 'cpu' # (Rebinning) String ('cpu','gpu','py') of function type - cpu, gpu, python, respectively - used to compute tomogram (3D). Defauts to 'cpu'.
dic['gpus'] = [0] # (Rebinning) List of GPU devices used for computation. GPU function uses only ONE GPU.

# =========== IMPORTANT =============================
# Order of Coneradon output: [slices,angles,X]
# Order of Rebinning input and output: [angles,slices,X]

# ======== Generate Conical Tomogram ===========================================
import time
start_ = time.time()

conesin = tomogram( phantom, dic, ttype )

print('Time for a phantom of', phantom.shape, 'shape,', dic['nangles'], 'angles,', dic['RayPoints'], 'integration points and', dic['Interpolation'], 'interpolation:', time.time() - start_)

# Changes [slices,angles,X] to [angles,slices,X]
conesin = np.swapaxes(conesin,0,1)

# ============ Generate Parallel Rebinning ================================
start_ = time.time()

parsin = rb.conebeam_rebinning_to_parallel(conesin, dic)

print('Time for a rebinning with', dic['Type'], 'function is', time.time() - start_)