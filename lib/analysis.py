import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
import lib.constants as c
from lib.parse_plot_data import import_dict
from lib.util import get_nearest_in_dataframe
from scipy.optimize import curve_fit
from scipy.special import erf


def conversion_atoms(delta, v_out, v_out_err):
    scat_rate = c.GAMMA / 2 * (c.INTENSITY / c.INTENSITY_SAT) / (
            1 + (c.INTENSITY / c.INTENSITY_SAT) + 4 * (delta / c.GAMMA) ** 2)
    energy = const.h * const.c / c.LASER_LENGTH
    atoms = v_out / (c.QE * c.G * c.S * c.T * c.SOLID_ANGLE * scat_rate * energy)
    err_atoms = np.sqrt((v_out_err / v_out) ** 2 + c.G_REL_ERROR ** 2 + c.QE_REL_ERROR ** 2 + ((c.INTENSITY_ERR/c.INTENSITY_SAT)/(
            1 + (c.INTENSITY / c.INTENSITY_SAT) + 4 * (delta / c.GAMMA) ** 2)-c.INTENSITY_ERR/c.INTENSITY)**2) * atoms
    return atoms, err_atoms


def loading_analysis(fit_df):
    # exclude the smallest detuning (no fit possible)
    fit_df = fit_df.drop(['TEK0006'])

    det = import_dict(c.DETUNING_DICT)
    det_arr = []
    for i in range(0, len(fit_df.index.values)):
        det_arr = np.append(det_arr, det[fit_df.index.values[i]])

    # errors
    alpha = 1 / fit_df['tau'].values
    err_alpha = fit_df['tau_err'].values / fit_df['tau'].values * alpha
    N, err_N = conversion_atoms(det_arr * 1e6, fit_df['amp'].values, fit_df['amp_err'].values)
    print('number of atoms:', N, 'error of number of atoms', err_N)
    L = np.multiply(alpha, N)
    err_L = np.sqrt((err_alpha / alpha) ** 2 + (err_N / N) ** 2) * L
    print('loading rate:', L, 'error of loading rate', err_L)

    fig, axis = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axis[0].errorbar(det_arr, L, xerr=c.DETUNING_ERROR, yerr=err_L, fmt='.', color=c.BLUE)
    axis[1].errorbar(det_arr, alpha, xerr=c.DETUNING_ERROR, yerr=err_alpha, fmt='.', color=c.RED)
    axis[2].errorbar(det_arr, N, yerr=err_N, xerr=c.DETUNING_ERROR, fmt='.', color=c.GREEN)
    axis[2].set(xlabel='$detuning \ [MHz]$', ylabel='$N$', title='$particle \ number$')
    axis[1].set(ylabel=r'$ \alpha \ [1/s]$', title='$loss \ coefficient$')
    axis[0].set(ylabel='$L \ [1/s]$', title='$loading \ rate$')

    plt.tight_layout()


def calculate_mean(fit_df, dfs):
    t0 = fit_df['t0'].values
    dt = fit_df['dt'].values

    mean1 = []
    err1 = []
    mean2 = []
    err2 = []
    for i, df in enumerate(dfs):
        mint0 = df.index.get_loc(t0[i] - 0.04, method='nearest')
        maxt0 = df.index.get_loc(t0[i], method='nearest')
        mindt = df.index.get_loc(dt[i] + 0.0001, method='nearest')
        maxdt = df.index.get_loc(dt[i] + 0.002, method='nearest')
        mean1 = np.append(mean1, np.mean(df.iloc[mint0:maxt0].values))
        err1 = np.append(err1, np.std(df.iloc[mint0:maxt0].values) / np.sqrt(len(df.iloc[mint0:maxt0].values)))
        mean2 = np.append(mean2, np.mean(df.iloc[mindt:maxdt].values))
        err2 = np.append(err2, np.std(df.iloc[mindt:maxdt].values) / np.sqrt(len(df.iloc[mindt:maxdt].values)))
    mean = [mean1, mean2, err1, err2]
    return mean


def recapture_analysis(mean):
    # Remove offset
    mean1 = mean[0] - mean[1][-6]
    mean2 = mean[1] - mean[1][-6]
    err1 = np.sqrt(mean[2] ** 2 + mean[3][-6] ** 2)
    err2 = np.sqrt(mean[3] ** 2 + mean[3][-6] ** 2)
    frac = mean2 / mean1
    err_frac = np.zeros(len(frac))
    for i in range(0, len(frac)):
        if frac[i] != 0:
            err_frac[i] = np.sqrt((err1[i] / mean1[i]) ** 2 + (err2[i] / mean2[i]) ** 2) * frac[i]

    duration = list(import_dict(c.DURATION_DICT).values())

    # Fitting the theoretical model
    def fitfunction(x, a):
        return erf(a / x) - 2 / np.sqrt(np.pi) * (a / x) * np.exp(-(a / x) ** 2)

    popt, pcov = curve_fit(fitfunction, duration, frac)
    param = popt[0]
    err_param = np.sqrt(np.diag(pcov))[0]
    temp = (c.MOT_RADIUS / param) ** 2 * c.RB85_MASS * 1e6 / const.k  # Temperature in K
    err_temp = err_param / param * temp
    print('temperature in K: {}, error: {}'.format(temp, err_temp))

    # Plot data
    plt.errorbar(duration, frac, yerr=err_frac, fmt='.', label='data point', color=c.BLUE)
    plt.plot(duration, fitfunction(duration, popt[0]), label='fit model', color=c.RED)
    plt.legend()
    plt.xlabel('$down \ time \ [ms]$')
    plt.ylabel('$N/N_0$')
    plt.text(68, 0.85, 'temperature = ' + str(int(round((temp * 1e6), 0))) + '$ \pm $' + str(
        int(round(err_temp * 1e6, 0))) + ' $\mu$K')


def save_temp_from_finestructure_in_fit_df(fit_data):
    sig = fit_data[['sig', 'sig_err']]
    constants = {'85f2': (c.RB85_MASS, c.RB85_NU0 - 1.77084), '85f3': (c.RB85_MASS, c.RB85_NU0 + 1.26489),
                 '87f1': (c.RB87_MASS, c.RB87_NU0 - 4.27168), '87f2': (c.RB87_MASS, c.RB87_NU0 + 2.56301)}
    temps = {}

    for index, row in sig.iterrows():
        mass, nu0 = constants[index]

        fwhm = 2 * row['sig'] * 2 * np.sqrt(2 * np.log(2))
        fwhm_err = fwhm * row['sig_err'] / row['sig']
        doppler_temp = fwhm * c.H_BAR / (4 * c.K_BOLTZMANN) * 1e9
        doppler_temp_err = doppler_temp * fwhm_err / fwhm
        fwhm_therm_gas = 293.15 * 2 * c.K_BOLTZMANN / c.H_BAR * 1e-9

        temp = (row['sig'] / nu0) ** 2 * mass * c.C ** 2 * (1 / (c.K_BOLTZMANN))
        temp_err = temp * (np.sqrt(2) * row['sig_err'] / row['sig'])
        sigma_therm_gas = nu0 * np.sqrt(c.K_BOLTZMANN * 293.15 / (mass * c.C ** 2))
        temps[index] = [temp, temp_err, sigma_therm_gas, fwhm, fwhm_err, doppler_temp, doppler_temp_err, fwhm_therm_gas]

    temp = pd.DataFrame.from_dict(data=temps, orient='index',
                                  columns=['Temperature atomic sample [K]',
                                           'error temperature [K]',
                                           'sigma thermal gas [GHz]',
                                           'FWHM [GHz]',
                                           'FWHM_err [GHz]',
                                           'Doppler temperature [K]',
                                           'Doppler temperature error [K]',
                                           'FWHM theory thermal gas (293.15 K) [GHz]'])

    fit_data = pd.concat([fit_data, temp], axis=1)
    return fit_data


def PDH_zero_crossings(dfs, guessed_zero_crossings):
    calculated_zero_crossings = {}
    count = 0
    guessed_zero_crossings = np.asarray(guessed_zero_crossings)

    for df in dfs:
        df, file_name = df
        zero_crossings = PDH_single_df_zero_crossing(df, guessed_zero_crossings[count])
        calculated_zero_crossings['{}'.format(count)], calculated_zero_crossings['{}_err'.format(count)]\
            = zero_crossings
        count += 1
    print('zero crossings from PDH signal', calculated_zero_crossings)
    return calculated_zero_crossings


def PDH_single_df_zero_crossing(df, guessed_zero_crossings):
    calculated_crossings = []
    calculated_crossings_error = []
    for guessed_crossing in guessed_zero_crossings:
        index = df.index.get_loc(guessed_crossing, "nearest")
        cropped_df = df.iloc[index-c.ZERO_CROSSING_SEARCH_OFFSET
                             :index+c.ZERO_CROSSING_SEARCH_OFFSET].loc[:, 'PDH out [a.u.]']
        y = cropped_df.values
        x = cropped_df.index.values
        single_crossing, crossing_error = single_zero_crossing(x, y)
        calculated_crossings.append(single_crossing)
        calculated_crossings_error.append(crossing_error)
    return (calculated_crossings, calculated_crossings_error)


def single_zero_crossing(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    crossing = x[c.ZERO_CROSSING_SEARCH_OFFSET - 1]
    crossing_err = 0
    for i in range(len(x)):
        if y[i-1] <= 0 and y[i] > 0:
            crossing = (x[i-1] + x[i]) / 2
            crossing_err = (x[i] - x[i-1]) / 2

    return (crossing, crossing_err)