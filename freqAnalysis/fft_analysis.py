import sys
sys.path.append("..")

from rwDat import read_data as rd
import numpy as np
import matplotlib.pyplot as plt


def fft_plot(y, dt, legend):
    """Plots the fft of the signal y"""
    N = 2**12
    yf = np.fft.fft(y, N)
    xf = np.fft.fftfreq(N, dt)
    yf_abs = np.abs(yf[:N//2])
    yf_nom = yf_abs/np.max(yf_abs)
    plt.plot(xf[:N//2], yf_nom, label=legend)

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Load the data
    test_data = rd.load_test_data_set(normalize=True)
    truck_data = rd.load_truck_data_set(normalize=True)

    # Plotting the fft
    plt.figure()
    for i in range(4):
        for j in range(2):
            fft_plot(truck_data[j][i].iod['u1'], truck_data[j][i].dt, truck_data[j][i].name + "_u1")
    # for i in range(4):
        # fft_plot(truck_data[1][i].iod['u1'], truck_data[0][i].dt, truck_data[0][i].name + "_u1")

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.xlim(-0.01, 0.15)
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.title('FFT of the NOx In signal')
    plt.savefig('./figs/fft_NOx_in_Truck.png')
    plt.show()
