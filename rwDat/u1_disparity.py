import numpy as np
import read_data as rd
import matplotlib.pyplot as plt
import scipy.signal as sig


# Acutaly load the entire data set -----------------------------------------
test_data = rd.load_test_data_set(normalize=True)
truck_data = rd.load_truck_data_set(normalize=True)

b = sig.butter(10, 0.01, 'low', fs = 1, output='sos')
# Plotting u1 form all the data sets
# for i in range(2):
#     for j in range(3):
#         plt.figure()
#         plt.plot(test_data[i][j].ssd['t'], test_data[i][j].ssd['u1'], label=test_data[i][j].name)
for i in range(2):
    for j in range(4):
        plt.figure()
#        plt.scatter(truck_data[i][j].iod['t'], sig.sosfiltfilt(b, truck_data[i][j].iod['u1']), label=truck_data[i][j].name, marker='.')
        plt.plot(truck_data[i][j].iod['t'], truck_data[i][j].iod['u1'], label=truck_data[i][j].name)
        plt.legend()
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('u1 [ppm]')
plt.show()
