### Useful functions 
import numpy as np
import scipy as sp


#import warnings


### The sigmoid function, used for spectral filtering
def sigmoid(cutoff, width, direction=-1, amplitude=1,
            cutoff_end=None, width_end=None,symmetric=True):
    if symmetric:
        sym = lambda s : np.absolute(s)
    else:
        sym = lambda s : s
        
    if cutoff_end is not None:
        if width_end is None: width_end=width
        return lambda s: amplitude / ( 1 + np.exp(-np.sign(direction)*(sym(s)-cutoff)/width) ) \
            / ( 1 + np.exp(np.sign(direction)*(sym(s)-cutoff_end)/width_end) )
    else:
        return lambda s: amplitude / ( 1 + np.exp(-np.sign(direction)*(sym(s)-cutoff)/width) )



### Exponential cosine, generates highly peaked, periodig curves
def expCos(k0, amplitude=1, peaked=None, FWHM=None, phi=np.pi):
    if peaked is None:
        peaked = np.log(2.) / ( 1 - np.cos(k0*0.5*FWHM) )
        
    if callable(amplitude):
        return lambda x, t: amplitude(t) * np.exp( peaked * (np.cos(k0*x+phi) - 1) )
    else:
        return lambda x: amplitude * np.exp( peaked * (np.cos(k0*x+phi) - 1) )

