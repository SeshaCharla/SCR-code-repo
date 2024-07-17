import sys
sys.path.append("..")

from rwDat import read_data as rd
import numpy as np
import scipy.signal as sig


# Manipulating functions ------------------------------------------------------

def sliced_derivatives(rrd:dict, nd:int,
                       poly_ord:int, window_len:int, delta_t:float):
    """
    Calculate the derivatives of the Data
    Data: dictionary with the data
    tsk: arrya with indices of the time steps to skip
    poly_ord: order of the polynomial for the Savitzky-Golay filter
    window_len: length of the window for the Savitzky-Golay filter (odd)
    Returns a dictionary with the derivatives
    """
    tsk = rrd['t_skips']
    N = len(rrd['t'])
    if len(rrd.keys()) == 7: # If data is IOD
        keys = ['y1', 'u1', 'u2', 'T', 'F', 'fv']
    elif len(rrd.keys()) == 8: # If data is SSD
        keys = ['x1', 'x2', 'u1', 'u2', 'T', 'F', 'fv']
    dkeys = ['d' + key for key in keys]
    for key in dkeys:  # Creating zero matrices for the derivatives
        rrd[key] = np.zeros([nd+1, N])
    # Savitzky-Golay FIR filter
    def sg_diff(x, n):
        return sig.savgol_filter(x, window_length=window_len,
                                 polyorder=poly_ord, deriv=n, delta=delta_t,
                                 mode='nearest')
    for (key, dkey) in zip(keys, dkeys):    # Selecting a state
        for j in range(nd+1):               # Selecting a derivative order
            for i in range(len(tsk)-1):     # Selecting a time slice
                if tsk[i+1] - tsk[i] > window_len:
                    rrd[dkey][j, tsk[i]:tsk[i+1]] = sg_diff(
                                                rrd[key][tsk[i]:tsk[i+1]], j)
                else:
                    raise ValueError("Window length is too samll run delSmallWindows() first")
    return rrd
# ------------------------------------------------------------------------------


def delSmallWindows(tab, tsk, window_len):
    """ Remove the data which is not continuous to the window length """
    def delFirstSmallWindow(tab, tsk, window_len):
        for i in range(len(tsk)-1):
            win = tsk[i+1] - tsk[i]
            if win <= window_len:
                tab = np.delete(tab, range(tsk[i], tsk[i+1]), axis=0)
                tsk[i+1:] = tsk[i+1:] - win
                tsk = np.delete(tsk, i)
                continue_flag = True
                return tab, tsk, continue_flag
        continue_flag = False
        return tab, tsk, continue_flag

    continue_flag = True
    while continue_flag:
        tab, tsk, continue_flag = delFirstSmallWindow(tab, tsk, window_len)
    return tab, tsk


class IdData(rd.Data):
    """Class for creating the identification data"""
    def __init__(self, tt, age, num, window_len=77, poly_ord=7):
        super().__init__(tt, age, num)
        self.normalize(rd.nom)
        self.window_len = window_len
        self.poly_ord = poly_ord
        self.delSmallWindows()
        self.gen_sliced_derivatives()

    def delSmallWindows(self):
        """ Remove the data which is not continuous to the window length """
        # SSD data
        if not (self.ssd is None):
            ssd_tab = np.matrix([self.ssd['t'], self.ssd['x1'], self.ssd['x2'],
                                 self.ssd['u1'], self.ssd['u2'],
                                 self.ssd['T'], self.ssd['F'],
                                 self.ssd['fv']]).T
            # Remove the data which is not continuous to the window length
            ssd_tab, self.ssd['t_skips'] = delSmallWindows(ssd_tab, self.ssd['t_skips'], self.window_len)
            ssd_mat = ssd_tab.T
            # Update the data
            self.ssd['t'] = np.array(ssd_mat[0]).flatten()
            self.ssd['x1'] = np.array(ssd_mat[1]).flatten()
            self.ssd['x2'] = np.array(ssd_mat[2]).flatten()
            self.ssd['u1'] = np.array(ssd_mat[3]).flatten()
            self.ssd['u2'] = np.array(ssd_mat[4]).flatten()
            self.ssd['T'] = np.array(ssd_mat[5]).flatten()
            self.ssd['F'] = np.array(ssd_mat[6]).flatten()
        # IOD data
        iod_tab = np.matrix([self.iod['t'], self.iod['y1'],
                             self.iod['u1'], self.iod['u2'],
                             self.iod['T'], self.iod['F']]).T
        # Remove the data which is not continuous to the window length
        iod_tab, self.iod['t_skips'] = delSmallWindows(iod_tab, self.iod['t_skips'], self.window_len)
        iod_mat = iod_tab.T
        # Update the data
        self.iod['t'] = np.array(iod_mat[0]).flatten()
        self.iod['y1'] = np.array(iod_mat[1]).flatten()
        self.iod['u1'] = np.array(iod_mat[2]).flatten()
        self.iod['u2'] = np.array(iod_mat[3]).flatten()
        self.iod['T'] = np.array(iod_mat[4]).flatten()
        self.iod['F'] = np.array(iod_mat[5]).flatten()

    def gen_sliced_derivatives(self):
        """ Generate the sliced derivative matrices """
        if not (self.ssd is None):
            self.ssd = sliced_derivatives(self.ssd, nd=4,
                                          poly_ord=self.poly_ord, window_len=self.window_len,
                                          delta_t=self.dt)
        self.iod = sliced_derivatives(self.iod, nd=4,
                                      poly_ord=self.poly_ord,
                                      window_len=self.window_len,
                                      delta_t=self.dt)


# ------------------------------------------------------------------------------

# Functions to load the Data sets ----------------------------------------------

def load_test_iddata():
    # Load the test Data
    test_data = [[IdData("test", age, tst) for tst in range(3)] for age in range(2)]
    return test_data


def load_truck_iddata():
    # Load the truck Data
    truck_data = [[IdData("truck", age, tst) for tst in range(4)] for age in range(2)]
    return truck_data

# ------------------------------------------------------------------------------


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    test_data = load_test_iddata()
    truck_data = load_truck_iddata()

    # Plotting the the derivatives
    dkeys_ssd = ['dx1', 'dx2', 'du1', 'du2', 'dF', 'dT', 'dfv']
    dkeys_iod = ['dy1', 'du1', 'du2', 'dF', 'dT', 'dfv']
    for age in test_data:
        for tst in age:
            for key in dkeys_ssd:
                for n in range(5):
                    plt.figure()
                    plt.plot(tst.ssd['t'], tst.ssd[key][n])
                    plt.xlabel('Time [s]')
                    plt.ylabel(key + '^(' + str(n)+')')
                    plt.title(tst.name + key + str(n))
                    plt.savefig('./figs/'+tst.name +'_ssd_'+ key + str(n) + '.png')
                    plt.close()
            for key in dkeys_iod:
                for n in range(5):
                    plt.figure()
                    plt.plot(tst.iod['t'], tst.iod[key][n])
                    plt.xlabel('Time [s]')
                    plt.ylabel(key + '^(' + str(n)+')')
                    plt.title(tst.name + key + str(n))
                    plt.savefig('./figs/'+tst.name + key +'_iod_'+ str(n) + '.png')
                    plt.close()

    for age in truck_data:
        for trk in age:
            for key in dkeys_iod:
                for n in range(5):
                    plt.figure()
                    plt.plot(trk.iod['t'], trk.iod[key][n])
                    plt.xlabel('Time [s]')
                    plt.ylabel(key + '^(' + str(n)+')')
                    plt.title(trk.name + key + str(n))
                    plt.savefig('./figs/'+trk.name + key +'_iod_'+ str(n) + '.png')
                    plt.close()
