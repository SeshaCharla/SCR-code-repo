import numpy as np
import read_data as rd
import matplotlib.pyplot as plt
import scipy.signal as sig


# Acutaly load the entire data set -----------------------------------------
test_data = rd.load_test_data_set()
truck_data = rd.load_truck_data_set()

# Plotting u1 form all the data sets
for i in range(2):
    for j in range(3):
        plt.figure()
        plt.plot(test_data[i][j].t, test_data[i][j].u1, label=test_data[i][j].name)
for i in range(2):
    for j in range(4):
        plt.figure()
        plt.plot(truck_data[i][j].t, sig.medfilt(truck_data[i][j].u1, kernel_size=201), label=truck_data[i][j].name)
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('u1 [ppm]')

plt.show()
