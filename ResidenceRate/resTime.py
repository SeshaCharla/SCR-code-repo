import sys
sys.path.append("..")

from rwDat import read_data as rd
from linIdent.identData import sliced_derivatives, delSmallWindows

import numpy as np


## SCR/ASC dimensions ----------------------------------------------------------
L_scr = 9.5*2.54 # cm
L_asc = 2*2.54 # cm
L = L_scr + L_asc # cm
D_scr = 13*2.54 # cm
A_cross = np.pi * (D_scr/2)**2 # cm^2
V_scr = A_cross * L # cm^3
## Density of exhaust gas ------------------------------------------------------
rho = 0.675e-3 # kg/cm^3

## Data class ------------------------------------------------------------------
class diffedData(rd.Data):
    """Class for creating the differentiable data"""
    def __init__(self, tt, age, num, window_len, poly_ord):
        super().__init__(tt, age, num)
        self.window_len = window_len
        self.poly_ord = poly_ord
        self.delSmallWindows()
        self.gen_sliced_derivatives()
        self.get_residence_time()
        self.get_NOxRedRate()

    def delSmallWindows(self):
        """ Remove the data which is not continuous to the window length """
        # SSD data
        if not (self.ssd is None):
            ssd_tab = np.matrix([self.ssd['t'], self.ssd['x1'], self.ssd['x2'],
                                 self.ssd['u1'], self.ssd['u2'],
                                 self.ssd['F'], self.ssd['T']]).T
            # Remove the data which is not continuous to the window length
            ssd_tab, self.ssd['t_skips'] = delSmallWindows(ssd_tab, self.ssd['t_skips'], self.window_len)
            ssd_mat = ssd_tab.T
            # Update the data
            self.ssd['t'] = np.array(ssd_mat[0]).flatten()
            self.ssd['x1'] = np.array(ssd_mat[1]).flatten()
            self.ssd['x2'] = np.array(ssd_mat[2]).flatten()
            self.ssd['u1'] = np.array(ssd_mat[3]).flatten()
            self.ssd['u2'] = np.array(ssd_mat[4]).flatten()
            self.ssd['F'] = np.array(ssd_mat[5]).flatten()
            self.ssd['T'] = np.array(ssd_mat[6]).flatten()
        # IOD data
        iod_tab = np.matrix([self.iod['t'], self.iod['y1'],
                             self.iod['u1'], self.iod['u2'],
                             self.iod['F'], self.iod['T']]).T
        # Remove the data which is not continuous to the window length
        iod_tab, self.iod['t_skips'] = delSmallWindows(iod_tab, self.iod['t_skips'], self.window_len)
        iod_mat = iod_tab.T
        # Update the data
        self.iod['t'] = np.array(iod_mat[0]).flatten()
        self.iod['y1'] = np.array(iod_mat[1]).flatten()
        self.iod['u1'] = np.array(iod_mat[2]).flatten()
        self.iod['u2'] = np.array(iod_mat[3]).flatten()
        self.iod['F'] = np.array(iod_mat[4]).flatten()
        self.iod['T'] = np.array(iod_mat[5]).flatten()

    def gen_sliced_derivatives(self):
        """ Generate the sliced derivative matrices """
        if not (self.ssd is None):
            self.ssd = sliced_derivatives(self.ssd, nd=1,
                                          poly_ord=self.poly_ord, window_len=self.window_len,
                                          delta_t=self.dt)
        self.iod = sliced_derivatives(self.iod, nd=1,
                                      poly_ord=self.poly_ord,
                                      window_len=self.window_len,
                                      delta_t=self.dt)

    def get_residence_time(self):
        """ Get the residence time """
        if self.ssd is None:
            F = self.iod['F'] * rd.kgmin2gsec_gain
        else:
            F = self.ssd['F'] * rd.kgmin2gsec_gain
        self.residence_time = (rho*V_scr)/(F)
        self.median_residence_time = np.median(self.residence_time)

    def get_NOxRedRate(self):
        """ Get the NOx reduction rate """
        if self.ssd is None:
            NOx_out = self.iod['dy1'][0] - np.median(self.iod['dy1'][0])
            dNOx_out = self.iod['dy1'][1]
            dNOx_in = self.iod['du1'][1]
        else:
            NOx_out = self.ssd['dx1'][0] - np.median(self.ssd['dx1'][0])
            dNOx_out = self.ssd['dx1'][1]
            dNOx_in = self.ssd['du1'][1]
        self.NOxRedRate = np.median(np.abs(dNOx_out - dNOx_in))
        self.reaction_band = self.NOxRedRate / np.median(np.abs(NOx_out))
        self.reaction_TimeConst = 1/self.reaction_band


# ------------------------------------------------------------------------------
# Functions to load the Data sets ----------------------------------------------
win_len = 101
poly_ord = 3

def load_test_iddata():
    # Load the test Data
    test_data = [[diffedData("test", age, tst, win_len, poly_ord) for tst in range(3)] for age in range(2)]
    return test_data


def load_truck_iddata():
    # Load the truck Data
    truck_data = [[diffedData("truck", age, tst, win_len, poly_ord) for tst in range(4)] for age in range(2)]
    return truck_data

# ------------------------------------------------------------------------------


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    test_data = load_test_iddata()
    truck_data = load_truck_iddata()

    # Plotting the residence times and the NOx reduction rates
    for age in test_data:
        for tst in age:
            plt.figure()
            plt.plot(tst.ssd['t'], tst.residence_time, label='Residence Time')
            plt.plot(tst.ssd['t'], [tst.median_residence_time]*np.ones(len(tst.ssd['t'])), label='Median Residence Time = {} s'.format(np.round(tst.median_residence_time, 2)))
            plt.plot(tst.ssd['t'], tst.dt*np.ones(len(tst.ssd['t'])), label='Sampling Time = {} s'.format(np.round(tst.dt, 2)))
            plt.xlabel('Time [s]')
            plt.ylabel('Timing [s]')
            plt.ylim(-0.1, 1.5)
            plt.title(tst.name)
            plt.legend()
            plt.grid()
            plt.savefig('./figs/'+tst.name + '_timing_stuff.png')
            plt.close()
            #===================================================================
#             plt.figure()
#             plt.plot(tst.ssd['t'], [tst.reaction_TimeConst]*np.ones(len(tst.ssd['t'])), label='Median Reaction Time Constant = {} s'.format(np.round(tst.reaction_TimeConst, 2)))
#             plt.xlabel('Time [s]')
#             plt.ylabel('Time constant for NOx reduction [s]')
#             plt.title(tst.name + ' NOx Reduction Rate')
#             plt.legend()
#             plt.grid()
#             plt.savefig('./figs/'+tst.name + '_NOxRedRate.png')
#             plt.close()

    for age in truck_data:
        for trk in age:
            plt.figure()
            plt.plot(trk.iod['t'], trk.residence_time, label='Residence Time')
            plt.plot(trk.iod['t'], trk.median_residence_time*np.ones(len(trk.iod['t'])), label='Median Residence Time = {}'.format(np.round(trk.median_residence_time, 2)))
            plt.plot(trk.iod['t'], trk.dt*np.ones(len(trk.iod['t'])), label='Sampling Time = {} s'.format(np.round(trk.dt, 2)))
            plt.xlabel('Time [s]')
            plt.ylabel('Timing [s]')
            plt.ylim(-0.1, 1.5)
            plt.title(trk.name)
            plt.legend()
            plt.grid()
            plt.savefig('./figs/'+trk.name + '_timing_stuff.png')
            plt.close()
            # =================================================================
#             plt.figure()
#             plt.plot(trk.iod['t'], trk.reaction_TimeConst*np.ones(len(trk.iod['t'])), label='Median Reaction Time Constant = {} s'.format(np.round(trk.reaction_TimeConst, 2)))
#             plt.xlabel('Time [s]')
#             plt.ylabel('Time constant for NOx reduction [s]')
#             plt.title(trk.name + ' NOx Reduction Rate')
#             plt.legend()
#             plt.grid()
#             plt.savefig('./figs/'+trk.name + '_NOxRedRate.png')
#             plt.close()
