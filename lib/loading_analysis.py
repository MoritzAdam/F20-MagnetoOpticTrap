import matplotlib.pyplot as plt
import numpy as np


def loading_analysis(fit_df):
    # import detuning data
    det = {}
    with open('../data/part3_detuning.txt') as f:
        for line in f:
            (key, val) = line.split()
            det[key] = float(val)
    err_det = 0.06

    # exclude the smallest detuning (no fit possible)
    fit_df = fit_df.drop(['TEK0006'])
    print(fit_df)

    det_arr = []
    for i in range(0, len(fit_df.index.values)):
        det_arr = np.append(det_arr, det[fit_df.index.values[i]])

    fig, axis = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    axis[0].errorbar(det_arr, np.multiply(fit_df['tau'].values, fit_df['amp'].values), xerr=err_det, fmt='.', color='b')
    axis[1].errorbar(det_arr, 1/fit_df['tau'].values, xerr=err_det, fmt='.', color='r')
    axis[2].errorbar(det_arr, fit_df['amp'].values, xerr=err_det, fmt='.', color='g')
    axis[2].set(xlabel='detuning frequency [MHz]', ylabel='N', title='particle numbers')
    axis[1].set(ylabel=r'$ \alpha $ [1/s]', title='loss coefficient')
    axis[0].set(ylabel='L [1/s]', title='loading rate')
    return fig, axis