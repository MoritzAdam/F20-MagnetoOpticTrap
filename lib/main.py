import numpy as np
import matplotlib.pyplot as plt
import lib.constants as c
from lib.util import mask_dfs, get_multiplet_separation, get_definition_zero, get_multiplet_df
from lib.parse_plot_data import get_txt_csv, make_oscilloscope_df, \
    make_spectroscopy_df, plot_dfs, \
    plot_dfs_spectroscopy
from lib.filter_data import filter_loading, filter_zoomed_spectroscopy, \
    calibrate_voltage_to_freq_scale, filter_recapture
from lib.analysis import loading_analysis, recapture_analysis
from lib.fit_data import fit_loading_dfs
from lib.analysis import loading_analysis, calculate_mean
from lib.fit_data import fit_loading_dfs, fit_spectroscopy_dfs


def main():
    # Loading rates
    plt_style = ['detuning = ', c.DETUNING_DICT, ' Mhz',  # Title, 1: start, 2: link of the dict, 3: end
               'intensity [a.u.]',  # ylabel
               'time [s]']  # xlabel
    dfs = make_oscilloscope_df(c.loading_path)
    plot_dfs(dfs, plt_style)
    plt.show()

    # Filtering values that are equal to the minimum value and Fitting loading curves
    dfs = filter_loading(dfs, rolling=10)
    dfs_fit, fit_df = fit_loading_dfs(dfs, offset_on=False)
    plot_dfs(dfs_fit, plt_style)
    plt.show()

    # Analysis of the loading curves
    loading_analysis(fit_df)
    plt.show()

    # Release, Recapture
    plt_style_rr = ['down time = ', c.DURATION_DICT, ' ms',  # Title, 1: start, 2: link of the dict, 3: end
               'intensity [a.u.]',  # ylabel
               'time [s]']  # xlabel
    dfs_rr = make_oscilloscope_df(c.temperature_path)
    plot_dfs(dfs_rr, plt_style_rr)
    plt.show()

    # Filtering Release, Recapture data and Fit
    dfs_rr = filter_recapture(dfs_rr)
    dfs_fit_rr, fit_df_rr = fit_loading_dfs(dfs_rr, offset_on=True)
    mean = calculate_mean(fit_df_rr, dfs_rr)
    plot_dfs(dfs_fit_rr, plt_style_rr, recapture=mean)
    plt.show()

    # Analysis of the recapture experiment
    recapture_analysis(mean)
    plt.show()

"""
    # Loading spectroscopy data
    dfs_spec = make_spectroscopy_df(c.spectroscopy_path)
    #fig, axes = plot_dfs_spectroscopy(dfs_spec, max_column_number=3, plot_PDH_out=True, plot_fit=False)
    #plt.show()

    # Calibrating the frequency scale
    dfs_compl_spec = dfs_spec[8:]
    dfs_compl_spec = mask_dfs(dfs_compl_spec, all_masks=[[(-0.66, -0.61), (-0.41, -0.375),
                                                          (0.18, 0.23), (0.66, 0.7)]])

    params = [(-0.2, -0.4, -0.3, -0.1,
               -0.6, -0.4, 0.23, 0.7,
               0.1, 0.1, 0.1, 0.1,
               0.6, 0.6, 0.6, 0.6,
               0.7, 0.7,
               -0.004, -0.004)]

    dfs_compl_spec, fit_df_compl_spec = fit_spectroscopy_dfs(dfs_compl_spec,
                                                             fct='poly_gaussian',
                                                             all_init_params=params)

    fig, axes = plot_dfs_spectroscopy(dfs_compl_spec, max_column_number=1, plot_PDH_out=False, plot_fit=True)
    plt.show()

    calibration_factor, calibration_factor_err = get_multiplet_separation(fit_df_compl_spec, 1, 4)
    definition_zero, definition_zero_error = get_definition_zero(fit_df_compl_spec)
    print('calibration factor: {}, zero definition: {}'.format(calibration_factor, definition_zero))

    df_multiplet_sep = get_multiplet_df(fit_df_compl_spec)

    # Gaussian fits of the transitions (unzoomed)
    dfs_spec = dfs_spec
    dfs_spec = filter_zoomed_spectroscopy(dfs_spec, return_zoomed=False, return_entire=False)
    dfs_spec = calibrate_voltage_to_freq_scale(dfs_spec,
                                               calibration_factor=calibration_factor,
                                               definition_zero=definition_zero)
    dfs_spec = mask_dfs(dfs_spec, all_masks=[[(0., 3.25), (3.85, 4.05)],
                                             [(1.25, 1.7), (2.05, 2.3)],
                                             [(6.05, 6.45)],
                                             [(-0.6, -0.2), (1.2, 1.3)]])

    dfs_spec, fit_df_spec = fit_spectroscopy_dfs(dfs_spec, fct='gaussian')

    fig, axes = plot_dfs_spectroscopy(dfs_spec, max_column_number=2, plot_PDH_out=False, plot_fit=True)
    plt.show()
    print(fit_df_spec)
"""

if __name__ == '__main__':
    main()