import sys
sys.path.append("..")

from rwDat import read_data as rd
import numpy as np

def sliced_derivatives(data:rd.Data, tsk:dict, poly_ord:int=7, window_len:int=15):
    ### Calculate the derivatives of the Data
    # Data: Data object
    # tsk: dictionary with the indices of the Data where the task switches
    # poly_ord: order of the polynomial for the Savitzky-Golay filter
    # window_len: length of the window for the Savitzky-Golay filter
    # Returns a dictionary with the derivatives
    dkeys = ['dx1', 'dx2', 'dy1', 'du1', 'du2', 'dT', 'dF']
    for key in dkeys:
        data[key] = np.zeros([4, len(data['t'])])
    for i in range(1, len(data['t'])):
        if i in tsk:
            for key in dkeys:
                data[key][:,i] = data[key][:,i-1]
        else:
            dt = data['t'][i] - data['t'][i-1]
            for key in dkeys:
                data[key][:,i] = (data[key][:,i-1] + data[key][:,i])/2
            data['dx1'][:,i] = (data['x1'][:,i] - data['x1'][:,i-1])/dt
            data['dx2'][:,i] = (data['x2'][:,i] - data['x2'][:,i-1])/dt
            data['dy1'][:,i] = (data['y1'][:,i] - data['y1'][:,i-1])/dt
            data['du1'][:,i] = (data['u1'][:,i] - data['u1'][:,i-1])/dt
            data['du2'][:,i] = (data['u2'][:,i] - data['u2'][:,i-1])/dt
            data['dT'][:,i] = (data['T'][:,i] - data['T'][:,i-1])/dt
            data['dF'][:,i] = (data['F'][:,i] - data['F'][:,i-1])/dt
    return data


class iddata:
    ### Class to calculate the state derivatives of the system
    def __init__(self, data):
        self.data = data
        if not self.data.normalized:
            self.data.normalize(rd.nom)
        self.tsk = dict()
        self.tsk['ssd'] = [i for i in range(1, len(self.data.ssd['t']))
                           if self.data.ssd['t'][i] - self.data.ssd['t'][i-1] > 1.5*self.data.dt]
        self.tsk['iod'] = [i for i in range(1, len(self.data.iod['t']))
                           if self.data.iod['t'][i] - self.data.iod['t'][i-1] > 1.5*self.data.dt]
        # Derivative matrices
        dkeys_ssd = ['dx1', 'dx2', 'du1', 'du2', 'dT', 'dF']
        for key in dkeys_ssd:
            self.data.ssd[key] = np.zeros([4, len(self.data.ssd['t'])])
        dkeys_iod = ['dy1', 'du1', 'du2', 'dT', 'dF']
        for key in dkeys_iod:
            self.data.iod[key] = np.zeros([4, len(self.data.iod['t'])])
        # Savitzky-Golay FIR filter parameters
        self.poly_ord = 7
        self.window_len= 15
        self.sg_diff = lambda x, n: sig.savgol_filter(x,
                                                window_length=window_len,
                                                polyorder=poly_ord,
                                                deriv=n, delta=dt)
        self.sliced_diff = lambda x, n: self.sg_diff()




if __name__ == "__main__":

    dat = iddata(rd.dg_rmc)
