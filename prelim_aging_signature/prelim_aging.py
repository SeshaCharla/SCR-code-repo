import sys
sys.path.append("..")

from rwDat import read_data as rd
import numpy as np
from scipy.optimize import nnls

def calc_beta(x1, u1, F, T):
    """Calculates the prelimnary aging factor alpha"""
    N = len(x1)
    beta = np.zeros(N-1)
    for i in range(N-1):
        beta[i] = (T[i]+273.15)*F[i] * ((u1[i] - x1[i+1])/u1[i])
    return beta

def calc_alphas(beta, T, n):
    """Calculates the aging factor alpha"""
    N = len(T)
    alpha_1 = np.zeros([N-n-4, 1])
    alpha_0 = np.zeros([N-n-4, 1])
    for i in range(N-4-n):
        B = np.zeros((n, 1))
        Tm = np.ones((n, 2))
        for j in range(n):
            B[j, 0] = beta[i+n-j-1]
            Tm[j, 0] = T[i+n-j-1] + 273.15
        alpha = nnls(Tm, B.flatten())[0]
        alpha_1[i, 0] = alpha[0]
        alpha_0[i, 0] = alpha[1]
    return (alpha_0, alpha_1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mat_size = 1750
    # load the data
    # Aged RMC
    beta_age = calc_beta(rd.ag_rmc.ssd['x1'], rd.ag_rmc.ssd['u1'], rd.ag_rmc.ssd['F'], rd.ag_rmc.ssd['T'])
    alpha_age = calc_alphas(beta_age, rd.ag_rmc.ssd['T'], mat_size)
    # DG RMC
    beta_dg = calc_beta(rd.dg_rmc.ssd['x1'], rd.dg_rmc.ssd['u1'], rd.dg_rmc.ssd['F'], rd.dg_rmc.ssd['T'])
    alpha_dg = calc_alphas(beta_dg, rd.dg_rmc.ssd['T'], mat_size)
    t = rd.ag_rmc.ssd['t']

    plt.figure()
    plt.plot((alpha_age[0][:, 0]), label='Age')
    plt.plot((alpha_dg[0][:, 0]), label='DG')
    plt.grid()
    plt.title(r'$\alpha_0$')
    plt.legend()


    plt.figure()
    plt.plot((alpha_age[1][:, 0]), label='Age')
    plt.plot((alpha_dg[1][:, 0]), label='DG')
    plt.grid()
    plt.legend()
    plt.title(r'$\alpha_1$')
    plt.show()
