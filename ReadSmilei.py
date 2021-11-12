### Smilei post processing for PseudoSpectral
################################################################################
import gc
import numpy as np
from numpy.fft import fft, ifft
import happi
import joiful.C
#import scipy as sp
#import constants_SI as SI # SI constants
import h5py
import warnings

#from .useful_functions import sigmoid


def rolling_average(x, n_avg):
    return np.convolve(x, np.ones(n_avg), 'valid') / n_avg


def periodic_corr(x, y):
    # Periodic correlation, implemented using the FFT.
    # x and y must be real sequences with the same length.

    return np.real_if_close(ifft( fft(x).conj() * fft(y) ))
    #return np.real_if_close(ifft( fft(x) * fft(y).conj() ))

class ReadSmilei:
     
    def __init__(self, smilei_path, smilei_output='', ps_l0_SI=None,movingWindow=True):

        ##### Initializing attributes to None
        self.SPath                = None
        
        self.tR2fs                = None
        self.lR2um                = None

        self.fs2tR                = None
        self.um2lR                = None
        self.tR2fs                = None
        self.lR2um                = None

        self.sm_l0_SI             = None
        self.sm2ps_L              = None
        self.sm2ps_T              = None
        self.sm2ps_omega          = None
        self.sm2ps_dens           = None

        self.t_pumpPeak           = None

        self.movingWindow         = None
        self.t_start_movingwindow = None
        self.vx_movingwindow      = None

        self.N_cells              = None
        self.L_cells              = None
        self.dimensions           = None

        self.n0_e                 = None
        self.x_vacuum             = None
        
        ##### Initializing data attributes
        ## Binned diagnostics
        self.binned               = None
        self.binnedInit           = False
        self.n_binning            = None
        ## Tracked diagnostics
        self.tracked              = None
        self.trackedInit          = False
        self.species              = None
        self.polarization         = None
        ## Smilei saved step info
        self.timeSteps            = None
        self.times                = None
        self.N_times              = None
        ## Smilei bin axes
        self.x_bins               = None
        self.y_bins               = None
        self.z_bins               = None
        ## Smilei simulation box info
        self.nx_avg               = None
        self.Nx                   = None
        self.Lx                   = None
        self.dt                   = None
        ## Variables containing the data
        self.x_t                  = None
        self.wp_sq                = None

        ### PS data
        self.x_ps                 = None

        ## Opening the Smilei simulation with happi
        self.S = happi.Open(smilei_path,verbose=False)
        
        if self.S.valid:
            self.SPath = smilei_path
            self.__basic_init__(ps_l0_SI,movingWindow)
        else:
            try:
                self.__file_init__(smilei_path,ps_l0_SI)
            except FileNotFoundError:
                Warning(f'{smilei_path} does not lead to any valid Smilei simulation or file.')
                
        
    def __basic_init__(self,ps_l0_SI, movingWindow):

        ## Smilei to PS units
        try:
            self.sm_l0_SI = self.S.namelist.l0_SI
        except AttributeError:
            Warning('Could not find wavelength in Smilei simulation; using 800 nm.')
            self.sm_l0_SI = 800e-9
        if ps_l0_SI is not None:
            self.sm2ps_L     = self.sm_l0_SI/ps_l0_SI
            self.sm2ps_T     = self.sm2ps_L
            self.sm2ps_omega = 1./self.sm2ps_T
            self.sm2ps_dens  = self.sm2ps_omega**2

            
        t0_SI = self.sm_l0_SI / 299792458.
        self.fs2tR = 1.0e-15 /(t0_SI/(2*np.pi))
        self.um2lR = 1.0e-6  /(self.sm_l0_SI/(2*np.pi))
        self.tR2fs = 1./self.fs2tR
        self.lR2um = 1./self.um2lR


        ## Time of pump-pulse peak amplitde when entering.
        try:
            self.t_pumpPeak = self.S.namelist.tCent
        except AttributeError:
            self.t_pumpPeak = None

        ## Smilei moving window parameters
        self.movingWindow = movingWindow
        if self.movingWindow:
            try:
                self.t_start_movingwindow = self.S.namelist.MovingWindow.time_start
                self.vx_movingwindow = self.S.namelist.MovingWindow.velocity_x
            except AttributeError:
                self.t_start_movingwindow = 0.0
                self.vx_movingwindow = 0.0
        else:
            self.t_start_movingwindow = 0.0
            self.vx_movingwindow = 0.0
        
        ## Smilei simulation cell data
        self.N_cells    = self.S.namelist.Main.number_of_cells
        self.L_cells    = self.S.namelist.Main.cell_length
        self.dimensions = len(self.N_cells)
        self.dt         = self.S.namelist.Main.timestep
        self.Lx         = self.L_cells[0]*self.N_cells[0]

        ## Extracting the electron density
        try:
            electrons = self.S.namelist.Species['electron']
            self.n0_e = electrons.number_density.value
            self.x_vacuum = electrons.number_density.xvacuum
        except AttributeError:
            self.n0_e = self.S.namelist.n0_e
            self.x_vacuum = self.S.namelist.xvacuum
        else:
            self.n0_e = 1.
            self.x_vacuum = 0

        ### end basic_init

    
    ### Inint from file
    def __file_init__(self, fname,ps_l0_SI):
        
        f = h5py.File(fname, "r")

        try:
            self.SPath            = f['SPath'][()].decode("utf-8")
            self.n_binning        = f['n_binning'][()]
        except KeyError:
            self.SPath            = None
            self.n_binning        = None
            
        
        self.Lx                   = f['Lx'][()]
        self.Nx                   = f['Nx'][()]
        self.times                = f['times'][()]
        self.x_t                  = f['x_t'][()]
        self.wp_sq                = f['wp_sq'][()]

        self.N_cells              = f['N_cells'][()]
        self.L_cells              = f['L_cells'][()]
        self.dimensions           = f['dimensions'][()]
        self.dt                   = f['dt'][()]

        self.t_pumpPeak           = f['t_pumpPeak'][()]
        self.sm_l0_SI             = f['sm_l0_SI'][()]
        self.n0_e                 = f['n0_e'][()]
        self.x_vacuum             = f['x_vacuum'][()]

        self.t_start_movingwindow = f['t_start_movingwindow'][()]
        self.vx_movingwindow      = f['vx_movingwindow'][()]

        t0_SI = self.sm_l0_SI / 299792458.
        self.fs2tR = 1.0e-15 /(t0_SI/(2*np.pi))
        self.um2lR = 1.0e-6  /(self.sm_l0_SI/(2*np.pi))
        self.tR2fs = 1./self.fs2tR
        self.lR2um = 1./self.um2lR

        if ps_l0_SI is not None:
            self.sm2ps_L     = self.sm_l0_SI/ps_l0_SI
            self.sm2ps_T     = self.sm2ps_L
            self.sm2ps_omega = 1./self.sm2ps_T
            self.sm2ps_dens  = self.sm2ps_omega**2
        else:
            self.sm2ps_L = f['sm2ps_L'][()]
            if self.sm2ps_L is not None:
                self.sm2ps_T     = self.sm2ps_L
                self.sm2ps_omega = 1./self.sm2ps_T
                self.sm2ps_dens  = self.sm2ps_omega**2
        
    ### end file_init

    ######################################################################
    ######################################################################

    ### Function for calculating the shift due to the moving window
    def _movingWindowShift(self,n=None,t=None):
        t_start = self.t_start_movingwindow
        vx      = self.vx_movingwindow
        if n is not None:
            return vx * max(0, n*self.dt-t_start)
        elif t is not None:
            return vx * max(0, t-t_start)
        else:
            ValueError("A timestep `n` or time `t` must be supplied.")
            
    ######################################################################
    ########################## Binned profiles ###########################
    ######################################################################
    
    ### A function for initializing the profile to be extracted
    def initBinnedProfile(self, n_binning=None,nx_avg=None):
        if self.S.valid:
            self._initBinnedProfile(n_binning,nx_avg)
        else:
            Warning("Cannot extract Smilei data. This object initilaized from file.")
    ###
    def _initBinnedProfile(self, n_binning,nx_avg):
        #self.n_binning = n_binning
        if n_binning is not None:
            if self.n_binning != n_binning:
                Warning(f'Changing particle binner from {self.n_binning} to {n_binning}.')
                self.n_binning = n_binning
        elif self.n_binning is None:
            Warning("No particle binning nbr supplied, using 0.")
            self.n_binning = 0

        if nx_avg is not None:
            if self.nx_avg != nx_avg:
                Warning(f'Changing nx_avg from {self.nx_avg} to {nx_avg}.')
                self.nx_avg = nx_avg
        elif self.nx_avg is None:
            self.nx_avg = 1
            

        self.binned    = self.S.ParticleBinning(self.n_binning)
        self.x_bins    = self.binned.getAxis("moving_x")
        self.y_bins    = self.binned.getAxis("y")
        self.z_bins    = self.binned.getAxis("z")
        
        # self.y_bins=None; self.z_bins=None
        # if self.dimensions>1: self.y_bins = binned.getAxis("y")
        # if self.dimensions>2: self.z_bins = binned.getAxis("z")

        self.timeSteps = self.binned.getTimesteps()
        self.times     = self.binned.getTimes()
        self.N_times   = self.timeSteps.size
        
        self.binnedInit = True
    #

    ### Function for extracting on-axis profile of binned data
    def extractBinnedProfile(self, n_binning=None, nx_avg=None, envelope=None, max_off_axis=None):
        if self.S.valid:
            self._extractBinnedProfile(n_binning, nx_avg, envelope, max_off_axis)
        else:
            Warning("Cannot extract Smilei data. This object initilaized from file.")
    ###
    def _extractBinnedProfile(self, n_binning, nx_avg, envelope, max_off_axis):
        if not self.binnedInit:
            self._initBinnedProfile(n_binning,nx_avg)

        self.Nx = self.x_bins.size +1-self.nx_avg
        x_avg  = rolling_average(self.x_bins, self.nx_avg)
        
        self.x_t   = np.zeros( (self.N_times, self.Nx) )
        self.wp_sq = np.zeros( self.x_t.shape )

        
        for i in range(self.N_times):
            n = self.timeSteps[i]
            #t = self.times[i]

            x_shift = x_avg + self._movingWindowShift(n=n)
            x = x_shift % self.Lx
            x[np.logical_and(x==0., x_shift//self.Lx>=1)] = self.Lx
            i_sort = np.argsort(x)
            self.x_t[i,:] = x[i_sort]
            
            ## Binned data
            wp_sq  = self.binned.getData(timestep=n)[0]
            
            ## Averaging out the z direction
            if len(self.z_bins)>0:
                if max_off_axis is not None:
                    L_half = self.L_cells[2]*self.N_cells[2]*0.5
                    limits = L_half + np.ones([-1.,1.])*max_off_axis
                    i_lim  = np.logical_and(L_half-max_off_axis <= self.z_bins,
                                            self.z_bins <= L_half+max_off_axis)
                    wp_sq  = np.mean(wp_sq[:,:,i_lim], axis=2)
                else:
                    wp_sq  = np.mean(wp_sq, axis=2)
            ## Averaging out the y direction
            if len(self.y_bins)>0:
                if max_off_axis is not None:
                    L_half = self.L_cells[1]*self.N_cells[1]*0.5
                    limits = L_half + np.ones([-1.,1.])*max_off_axis
                    i_lim  = np.logical_and(L_half-max_off_axis <= self.y_bins,
                                            self.y_bins <= L_half+max_off_axis)
                    wp_sq  = np.mean(wp_sq[:,i_lim], axis=1)
                else:
                    wp_sq  = np.mean(wp_sq, axis=1)
            ## end transverse averages
            
            wp_sq  = rolling_average(wp_sq, self.nx_avg)
            
            if self._movingWindowShift(n=n) < self.x_vacuum:
                wp_sq[x_avg>0.95*self.Lx]  = self.n0_e
               
            if callable(envelope):
                s  = envelope(x_avg)
                wp_sq = s*wp_sq + (1-s)*.5*(wp_sq[0]+wp_sq[-1])    
            
            self.wp_sq[i,:] = wp_sq[i_sort]
        ## end for time steps
    ## end extractBinnedProfile

    
    ######################################################################
    ######################### Tracked particles ##########################
    ######################################################################
    
    ### A function for initializing the profile to be extracted
    def initProfileBinner(self, species='electron',polarization=None):
        if self.S.valid:
            self._initProfileBinner(species,polarization)
        else:
            Warning("Cannot extract Smilei data. This object initilaized from file.")
    ###
    def _initProfileBinner(self, species,polarization):
        self.n_binning = -1

        if polarization is None:
            if self.polarization is None:
                self.polarization = 0.0 # default to y-polariazation
        else:
            self.polarization = polarization
            if self.polarization is not None:
                Warning(f'Changing polarization from {self.polarization:0.2f} to {polarization:0.2f}.')

        self.tracked   = self.S.TrackParticles(species=species,sort=False, chunksize=8e8)
        self.timeSteps = self.tracked.getTimesteps()
        self.times     = self.tracked.getTimes()
        self.N_times   = self.timeSteps.size
        
        self.trackedInit = True
    #

    ### Function for extracting on-axis profile of binned data
    def profileBinner(self, species='electron', Nx=None, xLim=None, yLim=None,zLim=None,
                      polarization=None, envelope=None):
        if self.S.valid:
            self._profileBinner(species, Nx, xLim, yLim,zLim, polarization,envelope)
        else:
            Warning("Cannot extract Smilei data. This object initilaized from file.")
    ###
    def _profileBinner(self, species, Nx, xLim, yLim,zLim, polarization,envelope):
        if not self.trackedInit:
            self._initProfileBinner(species,polarization)

        ## Init spatial parameters
        self.Nx = Nx
        if xLim is None: xLim = np.array([0,self.Lx])
        dx_bin = (xLim[1]-xLim[0])/self.Nx

        ## Inint arrays
        self.x_t   = np.zeros( (self.N_times, self.Nx) )
        self.wp_sq = np.zeros( self.x_t.shape )
        
        for i in range(self.N_times):
            n = self.timeSteps[i]
            print(f'i = {i} (of {self.N_times})')

            xLim_shift = xLim + self._movingWindowShift(n=n)
            
            particles = self.tracked.getData(timestep=n);
            weights   = particles[n]['w']
            x_data    = particles[n]['x']
            px_data   = particles[n]['px']/self.S.namelist.Species[species].mass
            py_data   = particles[n]['py']/self.S.namelist.Species[species].mass
            pz_data   = particles[n]['pz']/self.S.namelist.Species[species].mass

            def f_dv_pol(w, px,py,pz):
                gamma  = np.sqrt(1 + px**2 + py**2 + pz**2)
                p_pol  = np.cos(self.polarization)*py + np.sin(self.polarization)*pz
                return w/gamma * ( 1 - (p_pol/gamma)**2 )

            mask = None
            if (yLim is not None) or (zLim is not None):
                mask = np.full_like(weights, True, dtype=bool)
                if yLim is not None:
                    mask *= (particles[n]['y']>yLim[0]) * (particles[n]['y']<yLim[1])
                if zLim is not None:
                    mask *= (particles[n]['z']>zLim[0]) * (particles[n]['z']<zLim[1])
            # end create mask
            
            wp_sq, x_bin = joiful.C.get_dist1D(x_data,f_dv_pol(weights,px_data,py_data,pz_data),
                                               xLim_shift,Nx,
                                               mask=mask, shift_coordinates=False)
            # wp_sq, x_bin = joiful.C.get_dist1D(x_data,weights,
            #                                    xLim_shift,Nx,
            #                                    mask=mask, shift_coordinates=False)
            
            x = x_bin % self.Lx
            x[np.logical_and(x==0., x_bin//self.Lx>=1)] = self.Lx
            i_sort = np.argsort(x)
            self.x_t[i,:] = x[i_sort]
            
            if callable(envelope):
                s  = envelope(x_avg)
                wp_sq = s*wp_sq + (1-s)*.5*(wp_sq[0]+wp_sq[-1])    
            # end if callable
            
            self.wp_sq[i,:] = wp_sq[i_sort] / dx_bin


            ## Handling a memory overload issue, since `happi` stores
            ## every recalled timestep.
            del self.tracked._rawData
            gc.collect()
            self.tracked._rawData = None
        ## end for time steps
    ## end _profileBinner

    
    ######################################################################
    ########################### Interpolations ###########################
    ######################################################################

    ## Wrapper function for time interpolation of plasma
    ## profiles.
    ##
    ## There are two options: 'stationary' or 'wake'; the first option
    ## ('stationary') is a simple interpolation in time, and the
    ## second option ('wake') does an estimate of the pofile's
    ## veloctiy and shifts the profile and then interpolates.
    ##
    ## The option 'wake' is default due to backward compatibility
    def timeInterp(self,t=None,t_ps=None, simulation_type='wake',
                   plasma_threshold=None, t_wake_formed=None, **kwargs):

        if simulation_type=='wake':
            return self._timeInterp_wake(t=t,t_ps=t_ps, plasma_threshold=plasma_threshold,
                                         t_wake_formed=t_wake_formed, **kwargs)
        elif simulation_type=='stationary':
            return self._timeInterp_stationary(t=t,t_ps=t_ps, 
                                               **kwargs)
        else:
            raise ValueError(f"Selected simulation_type='{simulation_type}' does not exist.")

  
    ## A function which interpolates the Smilei data from a wake field
    ## setup in time. This function estimates a propagation velocity
    ## using a periodic correlation measurement.
    def _timeInterp_wake(self,t=None,t_ps=None,
                         plasma_threshold=None, t_wake_formed=None, **kwargs):
        if self.wp_sq is None:
            self.extractBinnedProfile(**kwargs)
        #
        if (t_ps is None) and (t is None):
            Warning("No time coordinate supplied, must have either t or t_ps.")
        elif (t_ps is not None) and (t is not None):
            Warning("Both t and t_ps supplied, using t.")
        elif (t_ps is not None) and (t is None):
            t = t_ps / self.sm2ps_T
        # else: #do nothing

        if t <= self.times[0]:
            ## For times before first saved time, return first saved profile
            return self.x_t[0,:], self.wp_sq[0,:]
        elif t > self.times[-1]:
            ## For times after last saved time, return last saved profile
            return self.x_t[-1,:], self.wp_sq[-1,:]
        else:
            ## For intermediary times, we interpolate in time and
            ## space, by finding the velocity of the wake and
            ## interpolating the shape of the wake together with the
            ## translation due to velocity.
            i  = np.argmax(self.times >= t) - 1 # time index closest to requested time
            t1 = self.times[i]                 # closest time below t
            t2 = self.times[i+1]               # closest time above t
            dt = t2 - t1                       # 
            a  = (t2 - t) / (t2 - t1)          # intepolation coefficient

            ## Interpolate the (unshifted) x coordinates. (This should
            ## be unnecessary, since the x coordinates shouldn't
            ## change.)
            x_interp = a*self.x_t[i,:] + (1-a)*self.x_t[i+1,:]

            
            if t_wake_formed is None:
                t_wake_formed = np.inf
            else:
                t_wake_formed = t_wake_formed / self.sm2ps_T

            if plasma_threshold is None:
                threshold = self.n0_e*0.25
            else:
                threshold = plasma_threshold / self.sm2ps_omega**2
            #
            j_vacuum  = np.argmax(self.wp_sq[i+1,:] > threshold)
            xt_vacuum = self.x_t[i+1,j_vacuum]#x_interp[j_vacuum]

            # ## Position of the plasma front
            # xt_vacuum = self.x_vacuum - self._movingWindowShift(t=t)            
            # ## Corresponding index of plasma front
            # j_vacuum = np.argmax(x_interp>xt_vacuum)

            ## We calculate the periodic correlation of the two
            ## profiles. The maximum of this, should give the index
            ## shift corresponiding to the velocity of the wake.
            ##
            ## Using the higher power to give the peaks larger weight
            correleation = periodic_corr(self.wp_sq[i,j_vacuum:]**3,self.wp_sq[i+1,j_vacuum:]**3)
            
            ## Spatial indices
            j_c = np.argmax(x_interp/dt > 1)    # index corresponding  to speed of light
            j_v = np.argmax(correleation[:j_c]) # Finding index that best match the velocity
            v   = x_interp[j_v]/dt              # Computing the corresponding velocity
            
            if j_vacuum <= 0 or t > t_wake_formed:
                ## Rolling back the next data set by corresponding
                ## velocity index, and then interpolate
                wp_sq_0 = self.wp_sq[i,:].copy()
                
                if j_v>0:
                    wp_sq_0[-j_v:]=wp_sq_0[:j_v]
                
                wp_sq_interp = a*np.roll(wp_sq_0,j_v) + (1-a)*self.wp_sq[i+1,:]
                
                # wp_sq_interp[:j_vacuum] = self.wp_sq[i,:j_vacuum]
                # wp_sq_interp[-j_vacuum:] = self.wp_sq[i,-j_vacuum:]
                
                ## Interpolating x coordinates, and shifting due to velocity
                x_t_tmp   = a*self.x_t[i,:] + (1-a)*self.x_t[i+1,:] -a*v*dt
                ## Rolling the shift back onto the periodic domain
                x_t_interp = x_t_tmp % self.Lx
                ## Coorinates on Lx stay on Lx
                x_t_interp[np.logical_and(x_t_interp==0., x_t_tmp//self.Lx>=1)] = self.Lx
                ## Finding the sorted indices
                i_sort = np.argsort(x_t_interp)
                
                return x_t_interp[i_sort], wp_sq_interp[i_sort]
            
            else:
                jjj = np.s_[j_vacuum::]; l_period=self.Lx-xt_vacuum
                
                x_t_interp   = self.x_t[i,:].copy()
                x_t_interp[jjj] = a*self.x_t[i,jjj] + (1-a)*self.x_t[i,jjj] -a*v*dt
                
                x_t_tmp2 = x_t_interp[jjj] - xt_vacuum
                x_t_tmp2 = x_t_tmp2 % l_period 
                x_t_tmp2[np.logical_and(x_t_tmp2==0., x_t_interp[jjj]//l_period>=1)] = l_period
                x_t_tmp2 += xt_vacuum
                x_t_interp[jjj] = x_t_tmp2 
                i_sort = np.argsort(x_t_interp)

                wp_sq_0 = self.wp_sq[i,:].copy()
                wp_sq_interp = a*wp_sq_0 + (1-a)*self.wp_sq[i+1,:]

                if j_v>0 and j_vacuum>j_v:
                    #wp_sq_0[-j_v:]=threshold
                    wp_sq_0[-j_v:]=wp_sq_0[(j_vacuum-j_v):(j_vacuum)]
                
                wp_sq_interp[jjj] = a*np.roll(wp_sq_0[jjj],j_v) + (1-a)*self.wp_sq[i+1,jjj]

                wp_sq_interp = wp_sq_interp[i_sort]
                if j_v>0:
                    wp_sq_0 = wp_sq_0[i_sort]
                    wp_sq_interp[-j_v:]=np.mean(wp_sq_0[-j_v:])
                
                ## Returing the sorted x_t and wp_sq (with same sort).
                return x_t_interp[i_sort], wp_sq_interp#[i_sort]
        ## end if time
    ## end _timeInterp_wake
    

    ## A function which interpolates the Smilei data from a stationary
    ## setup in time. 
    def _timeInterp_stationary(self,t=None,t_ps=None,
                               **kwargs):
        if self.wp_sq is None:
            self.extractBinnedProfile(**kwargs)
        #
        if (t_ps is None) and (t is None):
            Warning("No time coordinate supplied, must have either t or t_ps.")
        elif (t_ps is not None) and (t is not None):
            Warning("Both t and t_ps supplied, using t.")
        elif (t_ps is not None) and (t is None):
            t = t_ps / self.sm2ps_T
        # else: #do nothing

        if t <= self.times[0]:
            ## For times before first saved time, return first saved profile
            return self.x_t[0,:], self.wp_sq[0,:]
        elif t > self.times[-1]:
            ## For times after last saved time, return last saved profile
            return self.x_t[-1,:], self.wp_sq[-1,:]
        else:
            ## For intermediary times, we interpolate in time and
            ## space, by finding the velocity of the wake and
            ## interpolating the shape of the wake together with the
            ## translation due to velocity.
            i  = np.argmax(self.times >= t) - 1 # time index closest to requested time
            t1 = self.times[i]                 # closest time below t
            t2 = self.times[i+1]               # closest time above t
            dt = t2 - t1                       # 
            a  = (t2 - t) / (t2 - t1)          # intepolation coefficient

            return a*self.x_t[i,:] + (1-a)*self.x_t[i+1,:], \
                a*self.wp_sq[i,:] + (1-a)*self.wp_sq[i+1,:]   
        ## end if time
    ## end _timeInterp_stationary
    

    ## Function to be used by the PS code, for evaluating wp_sq on the
    ## PS grid, at time t_ps. [All in PS units.]
    def xtInterpPS(self, x_ps,t_ps,**kwargs):
        ## First interpolate in time
        x_t, wp_sq = self.timeInterp(t_ps=t_ps,**kwargs)
        ## Then interpolate in space onto the PS mesh (with PS normalization)
        return np.interp(x_ps, x_t*self.sm2ps_L, wp_sq*self.sm2ps_omega**2)


    
    ######################################################################
    ############################ Saving data #############################
    ######################################################################

    ## Function for saving data to a hdf5 file.
    def saveSmileiData(self,fname='smilei_data.h5'):
        ## hdf5 file to be written to.
        f = h5py.File(fname, "w")

        f.create_dataset('SPath',                data=self.SPath.encode("utf-8"))            
        ## Binning parameters
        f.create_dataset('n_binning',            data=self.n_binning, dtype=np.uint32)
        f.create_dataset('Nx',                   data=self.Nx,        dtype=np.uint32)
        f.create_dataset('times',                data=self.times)
        f.create_dataset('N_times',              data=self.N_times)
        ## Binned profiles
        f.create_dataset('x_t',                  data=self.x_t)
        f.create_dataset('wp_sq',                data=self.wp_sq)
        ## Smilei simulation parameters
        f.create_dataset('N_cells',              data=self.N_cells)
        f.create_dataset('L_cells',              data=self.L_cells)
        f.create_dataset('dimensions',           data=self.dimensions)
        f.create_dataset('Lx',                   data=self.Lx)
        f.create_dataset('dt',                   data=self.dt)
        ## Smilei physics parameters
        f.create_dataset('t_pumpPeak',           data=self.t_pumpPeak)
        f.create_dataset('sm_l0_SI',             data=self.sm_l0_SI)
        f.create_dataset('n0_e',                 data=self.n0_e)
        f.create_dataset('x_vacuum',             data=self.x_vacuum)
        ## Smilei moving window
        f.create_dataset('t_start_movingwindow', data=self.t_start_movingwindow)
        f.create_dataset('vx_movingwindow',      data=self.vx_movingwindow)
        ## Unit conversion factor      
        f.create_dataset("sm2ps_L",              data=self.sm2ps_L)

        f.close()
    ## end saveSmilei
