import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


def detuning(file_names):
    # import detuning data
    det = {}
    with open('../data/part3_detuning.txt') as f:
        for line in f:
            (key, val) = line.split()
            det[key] = float(val)
    return det[file_names]


def conversion_atomnumbers(delta, lam, v_out, v_out_err):
    p = 54  # P_powermeter in mW
    gamma = 2*np.pi*6.07*1e6  # Hz
    i_sat = 4.1  # mW/cm^2
    i = 2*p/(np.pi*0.2**2)  # mW/cm^2
    scat_rate = gamma/2*(i/i_sat)/(1+(i/i_sat)+4*(delta/gamma)**2)
    energy = const.h*const.c/lam
    solid_angle = np.pi*25.4**2/(4*np.pi*150**2)
    trans = 0.96
    g = 4.75*1e+6
    qe = 0.52
    s = 1e6/(1e6+50)
    atoms = v_out/(qe*g*s*trans*solid_angle*scat_rate*energy)
    err_atoms = np.sqrt((v_out_err/v_out)**2+0.05**2+0.029**2)*atoms
    return atoms, err_atoms


def loading_analysis(fit_df):
    # exclude the smallest detuning (no fit possible)
    fit_df = fit_df.drop(['TEK0006'])

    det_arr = []
    for i in range(0, len(fit_df.index.values)):
        det_arr = np.append(det_arr, detuning(fit_df.index.values[i]))

    # errors
    err_det = 0.06
    alpha = 1/fit_df['tau'].values
    err_alpha = fit_df['tau_err'].values/fit_df['tau'].values*alpha
    N, err_N = conversion_atomnumbers(det_arr*1e6, 780.241*1e-9, fit_df['amp'].values, fit_df['amp_err'].values)
    L = np.multiply(alpha, N)
    err_L = np.sqrt((err_alpha/alpha)**2+(err_N/N)**2)*L

    fig, axis = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    axis[0].errorbar(det_arr, L, xerr=err_det, yerr=err_L, fmt='.', color='b')
    axis[1].errorbar(det_arr, alpha, xerr=err_det, yerr=err_alpha, fmt='.', color='r')
    axis[2].errorbar(det_arr, N, yerr=err_N, xerr=err_det, fmt='.', color='g')
    axis[2].set(xlabel='detuning frequency [MHz]', ylabel='N', title='particle numbers')
    axis[1].set(ylabel=r'$ \alpha $ [1/s]', title='loss coefficient')
    axis[0].set(ylabel='L [1/s]', title='loading rate')
    return fig, axis