### Python file for quick use
import numpy as np
import scipy as sp
from scipy import integrate as ode
from scipy import interpolate as interp
from scipy import fft

import h5py

from .useful_functions import *

# import matplotlib as mpl
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation

import warnings


class PseudoSpectral:

    def __init__(self, fname=None,
                 xx=None, L=None, N=None, v_wake=0,
                 pulse=None, wp_sq=None, E_init=None,
                 wp_sq_init=None, filter_1=None, filter_2=None,
                 killReflections=False):
        
        if fname is not None: 
            self.__file_init__(fname)
        else:
            self.__basic_init__(xx=xx, L=L, N=N, pulse=pulse,
                                v_wake=v_wake, wp_sq=wp_sq, E_init=E_init,
                                wp_sq_init=wp_sq_init, filter_1=filter_1, filter_2=filter_2,
                                killReflections=killReflections)
    ## end __init__
            
    def __basic_init__(self, xx=None, L=None, N=None, pulse=None,
                       v_wake=0, wp_sq=None, E_init=None,
                       wp_sq_init=None, filter_1=None, filter_2=None,
                       killReflections=False):

        ## Initialization of the output variable
        self.OUT = None
        self.t   = None
        self.nt  = None

        if (L is not None) and (N is not None):
            self.L = L  # Box size
            self.N = N  # Nbr of grid points
            ## We don't include the end point in x, since the domain is
            ## L-periodic.
            self.x = np.linspace(0,L,num=N, endpoint=False)
            if xx is not None:
                warnings.warn('Input `xx` specified, but using `L` and `N` instead.')
        elif xx is not None:
            self.x = xx         # Spatial grid
            self.N = xx.size    # nbr of grid points
            ## Exrapolate one step in x to get the full box size
            self.L = 2*x[-1]-x[-2]
        else:
            ## There must be a spatial domain
            raise Exception('No spatial domain given.')

        
        self.timeVaryingMedia = False
        self.homogeneousMedia = False
        self.movingMedia = False
        self.v_wake = v_wake

        self.wp_sq        = None # Array
        self.wp_sq_time   = None # Function
        self.wp_sq_scalar = None # Scalar
        
        if callable(wp_sq):
            try:
                tmp = wp_sq(self.x,0)
                self.timeVaryingMedia = True
                self.wp_sq_time = lambda t: wp_sq(self.x,t)
                self.wp_sq = self.wp_sq_time(0)
            except TypeError:
                self.wp_sq = wp_sq(self.x)
                if self.v_wake != 0:
                    self.movingMedia = True
                    self.wp_sq_time=lambda t: self.__advect(self.wp_sq,int(self.v_wake*t))
                else:
                    self.wp_sq_time = lambda t: self.wp_sq
        elif np.isscalar(wp_sq):
            self.homogeneousMedia = True
            self.wp_sq_scalar = wp_sq
            self.wp_sq = self.wp_sq_scalar * np.ones(self.N)
            self.wp_sq_time = lambda t: self.wp_sq
        elif wp_sq.size == self.N:
            self.wp_sq = wp_sq
            if self.v_wake != 0:
                self.movingMedia = True
                self.wp_sq_time=lambda t: self.__advect(self.wp_sq,int(self.v_wake*t))
            else:
                self.wp_sq_time = lambda t: self.wp_sq
            
        else:
            raise Exception('Invalid `wp_sq` supplied.')
            

        self.Nf   = int(np.ceil(N/2))  # Number of frequencies
        self.k0   = 2*np.pi / self.L   # The lowest wavenumber
        self.kmax = self.Nf * self.k0  # The highest wavenember
        ## The full (pos. and neg.) wavenumber array
        self.k    = fft.fftfreq(self.N,self.L/self.N) * 2*np.pi
        self.k_sq = self.k**2   # Wavenumber squared

        ## The time-evolution filter
        if filter_1 is not None:
            self.fltr_1 = self.__ifCallable(filter_1, self.k)
        else:
            self.fltr_1 = 1

        ## Anti-aliasing filter
        if filter_2 is not None:
            self.fltr_2 = self.__ifCallable(filter_2, self.k)
        else:
            self.fltr_2 = 1

        ## Pulse real waveform (vector potential)
        self.a = self.__ifCallable(pulse, self.x)
        self.killReflections = killReflections

        ## Fourier decomposition (spatial) of the pulse
        self.A = fft.fft(self.a)

        ## The initial first time derivative (electric field) is
        ## calculated using the plasma dispersion.
        if E_init is not None:
            self.E  = E_init
        elif wp_sq_init is not None:
            ## If initial omega_p is give, we use that.
            self.w_init = np.sqrt(wp_sq_init+self.k_sq)
        else:
            ## Else, we use the omega_p at the peak of the pulse.
            i_pulse = np.argmax(np.absolute(self.a))
            self.w_init = np.sqrt(self.wp_sq[i_pulse]+self.k_sq)
        
        ## The (temporal) frequencies should have the same sign as
        ## each corresponding wavenumber. Otherwise the pulse splits
        ## in two counter-propagating pulses.
        self.w_init[self.Nf:] *= -1
        ## First time derivative (= electric field)
        self.E  = -1j * self.w_init * self.A
    ### end __basic_init__



    ### Inint from file
    def __file_init__(self, fname):

        f = h5py.File(fname, "r")

        self.x = f['x'][()]
        self.L = f['L'][()]
        self.N = int(f['N'][()])

        self.Nf   = f['Nf'][()]
        self.k0   = f['k0'][()]
        self.kmax = f['kmax'][()]
        self.k    = fft.fftfreq(self.N,self.L/self.N) * 2*np.pi
        self.k_sq = self.k**2   # Wavenumber squared

        self.nt = f['nt'][()]
        self.t  = f['t'][()]
        self.y = np.empty((2*self.N, self.nt), dtype=np.complex128)
        self.y.real = f['y/real'][()]
        self.y.imag = f['y/imag'][()]


        self.timeVaryingMedia = False
        self.homogeneousMedia = False
        self.movingMedia      = False
        
        self.v_wake   = f['v_wake'][()]
        
        wp_sq = f['wp_sq'][()]

        if np.isscalar(wp_sq): # basically a scalar
            self.homogeneousMedia = True
            self.wp_sq_scalar = wp_sq
            self.wp_sq = self.wp_sq_scalar * np.ones(self.N)
            self.wp_sq_time = lambda t: self.wp_sq
        elif wp_sq.size == self.N:
            self.wp_sq = wp_sq
            if self.v_wake != 0:
                self.movingMedia = True
                self.wp_sq_time = lambda t: self.__advect(self.wp_sq,int(self.v_wake*t))
            else:
                self.wp_sq_time = lambda t: self.wp_sq    
        elif wp_sq.size == self.N*self.nt:
            self.timeVaryingMedia = True
            self.wp_sq = wp_sq[0,:]
            self.wp_sq_time = interp.interp1d(self.t,wp_sq,axis=0)
        else:
            raise Exception('Invalid `wp_sq` supplied.')
        
        
        self.fltr_1 = f['fltr_1'][()]
        self.fltr_2 = f['fltr_2'][()]

        
        f.close()

    ### end __file_init__




    
    
    ## Helper function which tests if `obj` is a function, and if so
    ## returns the function evaluated over `var`.
    def __ifCallable(self, obj, var):
        if callable(obj):
            return obj(var)
        else:
            return obj

    def __advect(self,vec,j):
        n=vec.size
        i=np.arange(n)
        return vec[(i-j)%n]

    ## Function that computes the derivatives of each k-component for
    ## a static (but not necessarily homogeneous) plasma frequency.
    def __odefun(self,t, y):
        A   = y[:self.N]
        dA  = y[self.N:]*self.fltr_1
        S = fft.fft(self.wp_sq * fft.ifft(A*self.fltr_2))
        ddA = (-S -self.k_sq*A)*self.fltr_1
        
        # if self.killReflections:
        #     omega = -np.imag(dA/A)
        #     gamma = 0.5
        #     mask  = np.heaviside(-self.k * omega, 0)
        #     dA   -= A*mask*gamma
        #     #ddA  -= mask*gamma*dA
        
        return np.append(dA,ddA)

    ## Function that computes the derivatives of each k-component with
    ## a static and homogeneous plasma frequency    
    def __odefun_homo(self,t, y):
        A   = y[:self.N]
        dA  = y[self.N:]*self.fltr_1
        S   = self.wp_sq_scalar * A
        ddA = (-S -self.k_sq*A)*self.fltr_1    
        return np.append(dA,ddA)

    ## Function that computes the derivatives of each k-component with
    ## a time-varying plasma frequency
    def __odefun_time(self,t, y):
        A   = y[:self.N]
        dA  = y[self.N:]*self.fltr_1
        S = fft.fft(self.wp_sq_time(t) * fft.ifft(A*self.fltr_2))
        ddA = (-S -self.k_sq*A)*self.fltr_1
        return np.append(dA,ddA)

    
    ## Function which solves the wave equation in the specified time
    ## interval, with the object specified pulse and plasma profiles.
    def propagatePulse(self, t_span, rtol=1e-4, **kwargs):
        y0 = np.append(self.A,self.E)
        ## The different types of plasma profiles
        if self.timeVaryingMedia or self.movingMedia:
            self.OUT = ode.solve_ivp(self.__odefun_time, t_span, y0,
                                     rtol=rtol, **kwargs)
        elif self.homogeneousMedia:
            self.OUT = ode.solve_ivp(self.__odefun_homo, t_span, y0,
                                     rtol=rtol, **kwargs)
        else:
            self.OUT = ode.solve_ivp(self.__odefun, t_span, y0,
                                     rtol=rtol, **kwargs)
        self.y  = self.OUT.y
        self.t  = self.OUT.t
        self.nt = self.OUT.t.size


    
    
    ## Function for returning the final waveform
    def getWaveForm(self,real=True,i=None,i_range=None):
        if i is not None:
            B=self.y[0:self.N,i]*self.fltr_2
        # elif i_range is not None:
        #     B=self.y[0:self.N,i_range]*self.fltr_2
        else:
            if i_range is None: i_range=np.arange(self.nt)
            B=(self.y[0:self.N,i_range].T * self.fltr_2).T
        
        if real:
            return fft.ifft(B,axis=0)
        else:
            return B

    ## Function for returning the local temporal frequencies
    def getFrequencies(self,i=None):
        if i is None:
            omega=-np.imag(self.y[self.N:,:]/self.y[:self.N,:])
        else:
            omega=-np.imag(self.y[self.N:,i]/self.y[:self.N,i])
        return omega



    ## Function for getting the field time evolution at a specific point
    def getTimeEvolution(self,ix=None,it_range=None,
                         x=None,t_span=None):
        
        if (ix is None) and (x is not None):
            ix = np.argmax(self.x>=x)
        else:
            ValueError("You must supply a value for `ix` or `x`.")

        if (it_range is None):
            if (t_span is not None):
                it_range = np.where(np.logical_and(self.t>=t_span[0],
                                                   self.t<=t_span[1]))
            else:
                it_range = np.arange(self.nt)
            
        # if it_range is not None:
        #     c = self.getWaveForm(real=True,i_range=it_range)
        # elif t_span is not None:
        #     i_min = np.argmax(self.t>=t_span[0])
        #     i_max = np.argmax(self.t<t_span[1])-1
        #     ## Note that argmax will return 0 if t_span[1] is beyond
        #     ## the range of self.t, and we thus get the desired
        #     ## behavior that i_max=-1 if that happens.
        #     c = self.getWaveForm(real=True,i_span=[i_min,i_max])
        # else:
        #     c = self.getWaveForm(real=True)
        # return c[ix,:]
        c = self.getWaveForm(real=True,i_range=it_range)
        return c[ix,:]
    
    ## Function for extracting the temporal frequency spectrum at a
    ## location `self.x[ix]` (or `x`), in the time span t_span[0] to
    ## t_span[1].
    def getFrequencySpectrum(self,ix=None,it_range=None,
                             x=None,t_span=None,
                             fft_envelope=None, return_omega=False):
        
        if (ix is None) and (x is not None):
            ix = np.argmax(self.x>=x)
        else:
            ValueError("You must supply a value for `ix` or `x`.")

        if (it_range is None):
            if (t_span is not None):
                it_range = np.where(np.logical_and(self.t>=t_span[0],
                                                   self.t<=t_span[1]))
            else:
                it_range = np.arange(self.nt)

        c = self.getTimeEvolution(ix,it_range)
        n = it_range.size

        if fft_envelope is not None:
            fft_envelope_array = self.__ifCallable(fft_envelope,self.t[it_range])
            try:
                C = fft.fft(c*fft_envelope_array)
            except ValueError:
                Warning("The size of fft_envelope does not fit the chosen time range. Proceeding without envelope.")
                C = fft.fft(c)
        else:
            C = fft.fft(c)

        t_length = self.t[it_range[-1]] - self.t[it_range[0]]
        omega=fft.fftfreq(n,t_length/n) * 2*np.pi

        if return_omega:
            return omega, C
        else:
            return C









        

    ## Function for saving data to a hdf5 file.
    def saveData(self,fname='spectral_data.h5'):
        
        f = h5py.File(fname, "w")

        f.create_dataset('x', data=self.x)
        f.create_dataset('L',  data=self.L)
        f.create_dataset('N',  data=self.N)

        f.create_dataset('Nf',   data=self.Nf)
        f.create_dataset('k0',   data=self.k0)
        f.create_dataset('kmax', data=self.kmax)
        #k = fft.fftfreq(self.N,self.L/self.N) * 2*np.pi

        f.create_dataset('v_wake', data=self.v_wake)

        if self.homogeneousMedia:
            f.create_dataset('wp_sq', data=self.wp_sq_scalar)
        elif self.timeVaryingMedia:
            dset_wpt = f.create_dataset('wp_sq', (self.nt, self.N) )
            for i in range(self.nt):
                dset_wpt[i,:] = self.wp_sq_time(self.t[i]) 
        else:
            f.create_dataset('wp_sq', data=self.wp_sq)
        
        
        f.create_dataset("fltr_1", data=self.fltr_1)
        f.create_dataset("fltr_2", data=self.fltr_2)

        f.create_dataset('t',  data=self.t)
        f.create_dataset('nt', data=self.nt)

        spect_grp = f.create_group('y')
        spect_grp.create_dataset('real',data=self.y.real)
        spect_grp.create_dataset('imag',data=self.y.imag)

        f.close()

        
    

            



