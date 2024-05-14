import sys
sys.path.append("..")

from rwDat import read_data as rd
import numpy as np
import scipy.signal as sig


def sliced_derivatives(rrd:dict, nd:int=4,
                       poly_ord:int=7, window_len:int=21, delta_t:float=0.2):
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
    keys = list(rrd.keys())[1:6]
    dkeys = ['d' + key for key in keys]
    for key in dkeys:  # Creating zero matrices for the derivatives
        rrd[key] = np.zeros([nd+1, N])
    # Savitzky-Golay FIR filter
    def sg_diff(x, n):
        return sig.savgol_filter(x, window_length=window_len,
                                 polyorder=poly_ord, deriv=n, delta=delta_t)
    for (key, dkey) in zip(keys, dkeys):    # Selecting a state
        print(key, dkey)
        for j in range(nd+1):               # Selecting a derivative order
            for i in range(len(tsk)-1):     # Selecting a time slice
                if tsk[i+1] - tsk[i] > window_len:
                    rrd[dkey][j, tsk[i]:tsk[i+1]] = sg_diff(
                                                rrd[key][tsk[i]:tsk[i+1]], j)
                else:
                    raise ValueError("Window length is too samll run delSmallWindows() first")
        print(rrd[dkey])
    return rrd


class IdData(rd.Data):
    """Class for creating the identification data"""
    def __init__(self, tt, age, num, window_len=21, poly_ord=7):
        super().__init__(tt, age, num)
        self.normalize(rd.nom)
        self.window_len = window_len
        self.poly_ord = poly_ord
        self.delSmallWindows()
        self.gen_sliced_derivatives()

    def delSmallWindows(self):
        """ Remove the data which is not continuous to the window length """
        pass

    def gen_sliced_derivatives(self):
        """ Generate the sliced derivative matrices """
        print(self.name)
        if not (self.ssd is None):
            print('ssd')
            self.ssd = sliced_derivatives(self.ssd, nd=4,
                                          poly_ord=self.poly_ord, window_len=self.window_len,
                                          delta_t=self.dt)
        print('iod')
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
    # truck_data = load_truck_iddata()
