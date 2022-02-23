import numpy
import matplotlib.pyplot as plt
import time

from sscPhantom import gear
from sscBst import backprojection
from sscRadon import radon
from sscPhantom import mario

phantom = mario.createMario(shape=512, noise=False, zoom=0.1) #0.08
phantom = phantom/phantom.max()

#############

start = time.time()

dic = {'gpu': [0,1,2,3], 'blocksize':16, 'nangles': 512}

tomop = radon.tomogram(phantom, dic, 'parallel')

elapsed = time.time() - start

print('Elapsed time for parallel tomogram (sec):', elapsed )

#############


