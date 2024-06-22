import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

t = np.arange(0, 10, 0.01)
dt = float(t[1] - t[0])
w = 2*np.pi
y = np.sin(w*t) + np.random.normal(0, 0.1, t.shape)
yf = sig.savgol_filter(y, window_length=51, polyorder=2, deriv=1, delta=dt, mode='nearest')
z = (w)*np.cos(w*t)
# zf = sig.savgol_filter(y, window_length=51, polyorder=3, deriv=1, delta=0.1, mode='nearest')

plt.figure()
plt.plot(t, y, label='y')
plt.plot(t, yf, label='dy_savgol')
plt.plot(t, z, label='dy')
plt.legend()
plt.show()
