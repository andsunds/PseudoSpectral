### Smilei post processing for PseudoSpectral
################################################################################
import happi
import numpy as np
#import scipy as sp
#import constants_SI as SI # SI constants
import warnings


def sigma(x,w,L=1):
    return np.exp(-1/(x*2/w)**2)*np.exp(-1/((L-x)*2/w)**2)

def rolling_average(x, n_avg):
    return np.convolve(x, np.ones(n_avg), 'valid') / n_avg


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
        ## Smilei saved step info
        self.timeSteps            = None
        self.times                = None
        self.N_times              = None
        ## Smilei bin axes
        self.x_bins               = None
        self.y_bins               = None
        self.z_bins               = None
        ## Smilei simulation box info
        self.Nx                   = None
        self.Lx                   = None
        ## Variables containing the data
        self.x_t                  = None
        self.wp_sq                = None

        if S.valid:
            self.__basic_init__()
        else:
            try:
                self.__file_init(smilei_path)
            except FileNotFoundError:
                Warning(f'{smilei_path} does not lead to any valid Smilei simulation or file.')
                
        
    def __basic_init__(self):
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

    ### Function for extracting on-axis profile of binned data
    def extractBinnedProfile(self, n_binning=0, nx_avg=1, envelope=None, max_off_axis=None):

        binned = self.S.ParticleBinning(n_binning)
        self.x_bins = binned.getAxis("moving_x")
        self.y_bins = binned.getAxis("y")
        self.z_bins = binned.getAxis("z")
        
        # self.y_bins=None; self.z_bins=None
        # if self.dimensions>1: self.y_bins = binned.getAxis("y")
        # if self.dimensions>2: self.z_bins = binned.getAxis("z")

        self.timeSteps = binned.getTimesteps()
        self.times     = binned.getTimes()
        self.N_times   = self.timeSteps.size

        self.Nx = self.x_bins.size +1-nx_avg
        self.Lx = self.x_bins[-1]
        
        x_avg  = rolling_average(self.x_bins, nx_avg)
        
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
            wp_sq  = binned.getData(timestep=n)[0]
            
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
            
            wp_sq  = rolling_average(wp_sq, nx_avg)
               
            if callable(envelope):
                s  = envelope(x_avg)
                wp_sq = s*wp_sq + (1-s)*.5*(wp_sq[0]+wp_sq[-1])    
            
            self.wp_sq[i,:] = wp_sq[i_sort]
        ## end for time steps
    ## end extractBinnedProfile

    # # TODO:
    # def interpolatedBinnedProfile(self,N_ps,L_ps):
    #     if self.wp_sq is None:
    #         self.extractBinnedProfile()
        



    

    ## Function for saving data to a hdf5 file.
    def saveSmileiData(self,fname='smilei_data.h5'):
        
        f = h5py.File(fname, "w")

        
        f.create_dataset('Lx',  data=self.Lx)
        f.create_dataset('Nx',  data=self.Nx)
        f.create_dataset('times',  data=self.times)
        f.create_dataset('N_times', data=self.N_times)
        f.create_dataset('x_t', data=self.x_t)
        f.create_dataset('wp_sq', data=self.wp_sq)

        f.create_dataset('N_cells',  data=self.N_cells)
        f.create_dataset('L_cells',  data=self.L_cells)
        f.create_dataset('dimensions',  data=self.dimensions)
        
        f.create_dataset('t_pumpPeak', data=self.t_pumpPeak)
        f.create_dataset('n0_e',  data=self.n0_e)

        
        f.create_dataset('t_start_movingwindow', data=self.t_start_movingwindow)
        f.create_dataset('vx_movingwindow', data=self.vx_movingwindow)
        
        
        f.create_dataset("sm2ps_L", data=self.sm2ps_L)


        # spect_grp = f.create_group('y')
        # spect_grp.create_dataset('real',data=self.y.real)
        # spect_grp.create_dataset('imag',data=self.y.imag)

        f.close()
    ## end saveSmilei
