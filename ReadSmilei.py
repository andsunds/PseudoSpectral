### Smilei post processing for PseudoSpectral
################################################################################
import happi
import numpy as np
from numpy.fft import fft, ifft
#import scipy as sp
#import constants_SI as SI # SI constants
import warnings


def rolling_average(x, n_avg):
    return np.convolve(x, np.ones(n_avg), 'valid') / n_avg


def periodic_corr(x, y):
    # Periodic correlation, implemented using the FFT.
    # x and y must be real sequences with the same length.

    return np.real_if_close(ifft( fft(x).conj() * fft(y) ))
    #return np.real_if_close(ifft( fft(x) * fft(y).conj() ))

class ReadSmilei:
     
    def __init__(self, smilei_path, ps_l0_SI=None):

        ## Opening the Smilei simulation with happi
        self.S = happi.Open(smilei_path)
        # self.smileiSimulation = self.S

        ##### Initializing attributes to None
        self.tR2fs                = None
        self.lR2um                = None

        self.sm_l0_SI             = None
        self.sm2ps_L              = None
        self.sm2ps_T              = None
        self.sm2ps_omega          = None
        self.sm2ps_dens           = None

        self.t_pumpPeak           = None

        self.t_start_movingwindow = None
        self.vx_movingwindow      = None

        self.N_cells              = None
        self.L_cells              = None
        self.dimensions           = None

        self.n0_e                 = None

        ##### Initializing data attributes
        self.binned               = None
        self.binnedInit           = False
        self.n_binning            = None
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
        ## Variables containing the data
        self.x_t                  = None
        self.wp_sq                = None

        ### PS data
        self.x_ps                 = None

        if self.S.valid:
            self.__basic_init__(ps_l0_SI)
        else:
            try:
                self.__file_init(smilei_path)
            except FileNotFoundError:
                Warning(f'{smilei_path} does not lead to any valid Smilei simulation or file.')
                
        
    def __basic_init__(self,ps_l0_SI):
        ## Smilei to SI units
        try:
            self.tR2fs = 1./self.S.namelist.fs
            self.lR2um = 1./self.S.namelist.um
        except AttributeError:
            self.tR2fs = 1.
            self.lR2um = 1.

        ## The unit conversion
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

        ## Time of pump-pulse peak amplitde when entering.
        try:
            self.t_pumpPeak = self.S.namelist.tCent
        except AttributeError:
            self.t_pumpPeak = None

        ## Smilei moving window parameters
        try:
            self.t_start_movingwindow = self.S.namelist.MovingWindow.time_start
            self.vx_movingwindow = self.S.namelist.MovingWindow.velocity_x
        except AttributeError:
            self.t_start_movingwindow = 0.0
            self.vx_movingwindow = 0.0

        ## Smilei simulatio cell data
        self.N_cells    = self.S.namelist.Main.number_of_cells
        self.L_cells    = self.S.namelist.Main.cell_length
        self.dimensions = len(self.N_cells)

        ## Extracting the electron density
        electrons = self.S.namelist.Species['electron']
        self.n0_e = electrons.number_density.value
    ### end basic_init

    
    ### Inint from file
    def __file_init__(self, fname):
        
        f = h5py.File(fname, "r")

        
        self.Lx = f['Lx'][()]
        self.Nx = int(f['Nx'][()])
        self.times = f['times'][()]
        self.x_t = f['x_t'][()]
        self.wp_sq = f['wp_sq'][()]

        self.N_cells = f['N_cells'][()]
        self.L_cells = f['L_cells'][()]
        self.dimensions = f['dimensions'][()]

        self.t_pumpPeak = f['t_pumpPeak'][()]
        self.n0_e = f['n0_e'][()]

        self.t_start_movingwindow = f['t_start_movingwindow'][()]
        self.vx_movingwindow = f['vx_movingwindow'][()]

        self.sm2ps_L = f['sm2ps_L'][()]
        if self.sm2ps_L is not None:
            self.sm2ps_T     = self.sm2ps_L
            self.sm2ps_omega = 1./self.sm2ps_T
            self.sm2ps_dens  = self.sm2ps_omega**2
        
    ### end file_init
    

    ### Function for calculating the shift due to the moving window
    def _movingWindowShift(self,n=None,t=None):
        t_start = self.t_start_movingwindow
        vx      = self.vx_movingwindow
        if n is not None:
            dt      = self.S.namelist.Main.timestep
            return vx * max(0, n*dt-t_start)
        elif t is not None:
            return vx * max(0, t-t_start)
        else:
            ValueError("A timestep `n` or time `t` must be supplied.")
            
    
    ### A function for initializing the profile to be extracted
    def initBinnedProfile(self, n_binning=None,nx_avg=None):
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

        self.Lx = self.x_bins[-1]
        
        self.binnedInit = True

    ### Function for extracting on-axis profile of binned data
    def extractBinnedProfile(self, n_binning=None, nx_avg=None, envelope=None, max_off_axis=None):
        if not self.binnedInit:
            self.initBinnedProfile(n_binning,nx_avg)

        self.Nx = self.x_bins.size +1-self.nx_avg
        x_avg  = rolling_average(self.x_bins, self.nx_avg)
        
        self.x_t   = np.zeros( (self.N_times, self.Nx) )
        self.wp_sq = np.zeros( self.x_t.shape )

        
        for i in range(self.N_times):
            n = self.timeSteps[i]
            #t = self.times[i]

            x_shift = x_avg + self._movingWindowShift(n=n)
            x = x_shift%self.Lx
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
               
            if callable(envelope):
                s  = envelope(x_avg)
                wp_sq = s*wp_sq + (1-s)*.5*(wp_sq[0]+wp_sq[-1])    
            
            self.wp_sq[i,:] = wp_sq[i_sort]
        ## end for time steps
    ## end extractBinnedProfile

    ##
    def initPSInterp(self,x_ps, ps_l0_SI=None):
        ## The unit conversion
        if ps_l0_SI is not None:
            Warning(f"Now using ps_l0_SI = {ps_l0_SI}.")
            self.sm2ps_L     = self.sm_l0_SI/ps_l0_SI
            self.sm2ps_T     = self.sm2ps_L
            self.sm2ps_omega = 1./self.sm2ps_T
            self.sm2ps_dens  = self.sm2ps_omega**2
        #
        self.x_ps = x_ps
    ## end initPSInterp

  
    ## A function which interpolates the Smilei data in time. This
    ## function estimates a propagation velocity using a periodic
    ## correlation measurement.
    def timeInterp(self,t=None,t_ps=None,**kwargs):
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
            i  = np.argmax(self.times > t) - 1 # time index closest to requested time
            t1 = self.times[i]                 # closest time below t
            t2 = self.times[i+1]               # closest time above t
            dt = t2 - t1                       # 
            a  = (t2 - t) / (t2 - t1)          # intepolation coefficient

            ## We calculate the periodic correlation of the two
            ## profiles. The maximum of this, should give the index
            ## shift corresponiding to the velocity of the wake.
            correleation = periodic_corr(self.wp_sq[i,:],self.wp_sq[i+1,:])
            ## Interpolate the (unshifted) x coordinates. (This should
            ## be unnecessary, since the x coordinates shouldn't
            ## change.)
            x_interp = a*self.x_t[i,:] + (1-a)*self.x_t[i+1,:]
            
            ## Spatial indices
            j_c = np.argmax(x_interp/dt > 1)    # index corresponding  to speed of light
            j_v = np.argmax(correleation[:j_c]) # Finding index that best match the velocity
            v   = x_interp[j_v]/dt              # Computing the corresponding velocity
            
            ## Rolling back the next data set by corresponding
            ## velocity index, and then interpolate
            wp_sq_interp = a*self.wp_sq[i,:] + (1-a)*np.roll(self.wp_sq[i+1,:],-j_v)
            ## Interpolating x coordinates, and shifting due to velocity
            x_t_tmp   = a*self.x_t[i,:] + (1-a)*self.x_t[i+1,:] + a*v*dt
            ## Rolling the shift back onto the periodic domain
            x_t_interp = x_t_tmp % self.Lx
            ## Coorinates on Lx stay on Lx
            x_t_interp[np.logical_and(x_t_interp==0., x_t_tmp//self.Lx>=1)] = self.Lx
            ## Finding the sorted indices
            i_sort = np.argsort(x_t_interp % self.Lx)
            ## Returing the sorted x_t and wp_sq (with same sort).
            return x_t_interp[i_sort], wp_sq_interp[i_sort]
        ## end if time
    ## end timeInterp

    ## Function to be used by the PS code, for evaluating wp_sq on the
    ## PS grid, at time t_ps. [All in PS units.]
    def xtInterpPS(self,t_ps):
        ## First interpolate in time
        x_t, wp_sq = self.timeInterp(t_ps=t_ps)
        ## Then interpolate in space onto the PS mesh (with PS normalization)
        return np.interp(self.x_ps, x_t*self.sm2ps_L, wp_sq*self.sm2ps_omega**2)

    
    

    ## Function for saving data to a hdf5 file.
    def saveSmileiData(self,fname='smilei_data.h5'):
        ## hdf5 file to be written to.
        f = h5py.File(fname, "w")

        ## Binning parameters
        f.create_dataset('Lx',  data=self.Lx)
        f.create_dataset('Nx',  data=self.Nx)
        f.create_dataset('times',  data=self.times)
        f.create_dataset('N_times', data=self.N_times)
        ## Binned profiles
        f.create_dataset('x_t', data=self.x_t)
        f.create_dataset('wp_sq', data=self.wp_sq)
        ## Smilei simulation parameters
        f.create_dataset('N_cells',  data=self.N_cells)
        f.create_dataset('L_cells',  data=self.L_cells)
        f.create_dataset('dimensions',  data=self.dimensions)
        ## Smilei physics parameters
        f.create_dataset('t_pumpPeak', data=self.t_pumpPeak)
        f.create_dataset('n0_e',  data=self.n0_e)
        ## Smilei moving window
        f.create_dataset('t_start_movingwindow', data=self.t_start_movingwindow)
        f.create_dataset('vx_movingwindow', data=self.vx_movingwindow)
        
        
        f.create_dataset("sm2ps_L", data=self.sm2ps_L)


        # spect_grp = f.create_group('y')
        # spect_grp.create_dataset('real',data=self.y.real)
        # spect_grp.create_dataset('imag',data=self.y.imag)

        f.close()
    ## end saveSmilei
