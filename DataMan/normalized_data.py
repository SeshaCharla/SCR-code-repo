import numpy as np
import read_data as rd

# Nominal values for the normalization
# Using RMC data to get the nominal values
# The RMC data is on an average close to the truck data even though the truck
# data has more frequent spikes
dg_rmc = rd.data("test", 0, 2)
ag_rmc = rd.data("test", 1, 2)
nom_vals = {'x1': np.mean([dg_rmc.Medians['x1'], ag_rmc.Medians['x1']]),
            'x2': np.mean([dg_rmc.Medians['x2'], ag_rmc.Medians['x2']]),
            'u1': np.mean([dg_rmc.Medians['u1'], ag_rmc.Medians['u1']]),
            'u2': np.mean([dg_rmc.Medians['u2'], ag_rmc.Medians['u2']]),
            'F': np.mean([dg_rmc.Medians['F'], ag_rmc.Medians['F']]),
            'T': np.mean([dg_rmc.Medians['T'], ag_rmc.Medians['T']]),
            'y1': np.mean([dg_rmc.Medians['x1'], ag_rmc.Medians['x1']])}


class normalized_data():
    def __init__(self, data):
        # Generate the normalized data
        x1 = (data.x1 - nom_vals['x1'])/nom_vals['x1']
        x2 = (data.x2 - nom_vals['x2'])/nom_vals['x2']
        u1 = (data.u1 - nom_vals['u1'])/nom_vals['u1']
        u2 = (data.u2 - nom_vals['u2'])/nom_vals['u2']
        F = (data.F - nom_vals['F'])/nom_vals['F']
        T = (data.T - nom_vals['T'])/nom_vals['T']
        y1 = (data.y1 - nom_vals['y1'])/nom_vals['y1']
        t = data.t
        self.name = data.name
        state_tab = rd.rmNaNrows(np.matrix([t, x1, x2, u1, u2, T, F]).T)
        output_tab = rd.rmNaNrows(np.matrix([t, y1, u1, u2, T, F]).T)
        self.state_mat = state_tab.T
        self.output_mat = output_tab.T


if __name__=="__main__":
    import matplotlib.pyplot as plt

    # Acutaly load the entire data set -----------------------------------------
    test_data = rd.load_test_data_set()
    # truck_data = rd.load_truck_data_set()

    test_data_norm = [[normalized_data(test_data[i][j]) for j in range(3)] for i in range(2)]
    # truck_data_norm = [[normalized_data(truck_data[i][j]) for j in range(4)] for i in range(2)]

    # Scatter plots of all the data sets
    for i in range(2):
        for j in range(3):
            t = np.array(test_data_norm[i][j].state_mat[0]).flatten()
            for k in range(1,7):
                plt.figure()
                plt.scatter(t, np.array(test_data_norm[i][j].state_mat[k]).flatten(), marker='.',linewidths=0.1, label=test_data[i][j].name)

plt.show()
