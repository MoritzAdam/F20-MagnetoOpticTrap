import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import lib.constants as c
from lib.parse_plot_data import import_dict
from scipy.optimize import curve_fit
from scipy.special import erf


def conversion_atoms(delta, v_out, v_out_err):
    scat_rate = c.GAMMA/2*(c.INTENSITY/c.INTENSITY_SAT)/(1+(c.INTENSITY/c.INTENSITY_SAT)+4*(delta/c.GAMMA)**2)
    energy = const.h*const.c/c.LASER_LENGTH
    atoms = v_out/(c.QE*c.G*c.S*c.T*c.SOLID_ANGLE*scat_rate*energy)
    err_atoms = np.sqrt((v_out_err/v_out)**2+c.G_REL_ERROR**2+c.QE_REL_ERROR**2)*atoms
    return atoms, err_atoms


def loading_analysis(fit_df):
    # exclude the smallest detuning (no fit possible)
    fit_df = fit_df.drop(['TEK0006'])

    det = import_dict(c.DETUNING_DICT)
    det_arr = []
    for i in range(0, len(fit_df.index.values)):
        det_arr = np.append(det_arr, det[fit_df.index.values[i]])

    # errors
    alpha = 1/fit_df['tau'].values
    err_alpha = fit_df['tau_err'].values/fit_df['tau'].values*alpha
    N, err_N = conversion_atoms(det_arr*1e6, fit_df['amp'].values, fit_df['amp_err'].values)
    L = np.multiply(alpha, N)
    err_L = np.sqrt((err_alpha/alpha)**2+(err_N/N)**2)*L

    fig, axis = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    axis[0].errorbar(det_arr, L, xerr=c.DETUNING_ERROR, yerr=err_L, fmt='.', color=c.BLUE)
    axis[1].errorbar(det_arr, alpha, xerr=c.DETUNING_ERROR, yerr=err_alpha, fmt='.', color=c.RED)
    axis[2].errorbar(det_arr, N, yerr=err_N, xerr=c.DETUNING_ERROR, fmt='.', color=c.GREEN)
    axis[2].set_xlabel('detuning [MHz]', style='italic')
    axis[2].set_ylabel('N', style='italic')
    axis[2].set_title('particle numbers', fontweight='semibold')
    axis[1].set_ylabel(r'$ \alpha $ [1/s]', style='italic')
    axis[1].set_title('loss coefficient', fontweight='semibold')
    axis[0].set_ylabel('L [1/s]', style='italic')
    axis[0].set_title('loading rate', fontweight='semibold')
    return fig, axis


def calculate_mean(fit_df, dfs):
    t0 = fit_df['t0'].values
    dt = fit_df['dt'].values

    mean1 = []
    err1 = []
    mean2 = []
    err2 = []
    for i, df in enumerate(dfs):
        mint0 = df.index.get_loc(t0[i]-0.04, method='nearest')
        maxt0 = df.index.get_loc(t0[i], method='nearest')
        mindt = df.index.get_loc(dt[i]+0.0001, method='nearest')
        maxdt = df.index.get_loc(dt[i]+0.002, method='nearest')
        mean1 = np.append(mean1, np.mean(df.iloc[mint0:maxt0].values))
        err1 = np.append(err1, np.std(df.iloc[mint0:maxt0].values)/np.sqrt(len(df.iloc[mint0:maxt0].values)))
        mean2 = np.append(mean2, np.mean(df.iloc[mindt:maxdt].values))
        err2 = np.append(err2, np.std(df.iloc[mindt:maxdt].values)/np.sqrt(len(df.iloc[mindt:maxdt].values)))
    mean = [mean1, mean2, err1, err2]
    return mean


def recapture_analysis(mean):
    # Remove offset
    mean1 = mean[0] - mean[1][-6]
    mean2 = mean[1] - mean[1][-6]
    err1 = np.sqrt(mean[2]**2+mean[3][-6]**2)
    err2 = np.sqrt(mean[3]**2+mean[3][-6]**2)
    frac = mean2/mean1
    err_frac = np.zeros(len(frac))
    for i in range(0, len(frac)):
        if frac[i] != 0:
            err_frac[i] = np.sqrt((err1[i]/mean1[i])**2+(err2[i]/mean2[i])**2)*frac[i]

    duration = list(import_dict(c.DURATION_DICT).values())

    # Fitting the theoretical model
    def fitfunction(x, a):
        return erf(a / x) - 2 / np.sqrt(np.pi) * (a / x) * np.exp(-(a / x) ** 2)

    popt, pcov = curve_fit(fitfunction, duration, frac)
    param = popt[0]
    err_param = pcov[0][0]
    temp = (c.MOT_RADIUS/param)**2*c.RB85_MASS*1e6/const.k  # Temperature in K
    err_temp = err_param/param*temp

    # Plot data
    plt.errorbar(duration, frac, yerr=err_frac, fmt='.', label='data point', color=c.BLUE)
    plt.plot(duration, fitfunction(duration, popt[0]), label='fit model', color=c.RED)
    plt.legend()
    plt.title('Analysis of the Release and Recapture experiment', fontweight='semibold')
    plt.xlabel('down time [ms]', style='italic')
    plt.ylabel('$N/N_0$', style='italic')
    plt.text(68, 0.85, 'temperature = '+str(int(round((temp*1e6),0)))+'$ \pm $'+str(int(round(err_temp*1e6,0)))+' $\mu$K')