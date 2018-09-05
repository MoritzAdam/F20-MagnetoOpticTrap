from lib.parse_plot_data import get_txt_csv, make_oscilloscope_df, \
    make_spectroscopy_df, plot_dfs, plot_dfs_with_fits, \
    plot_dfs_spectroscopy
from lib.filter_data import filter_loading, filter_zoomed_spectroscopy, \
    calibrate_voltage_to_freq_scale
from lib.fit_data import fit_loading_dfs, fit_spectroscopy_entire_dfs
from lib.loading_analysis import loading_analysis
from lib.fit_data import fit_loading_dfs, fit_spectroscopy_single_dfs
import matplotlib.pyplot as plt
import lib.constants as c
from lib.util import mask_dfs


def main():
    # Loading rates
    # 1 - Starting point
    #dfs = make_oscilloscope_df(c.loading_path)
    #fig, axes = plot_dfs(dfs)
    #plt.show()
#
    ## 2 - Filtering values that are equal to the minimum value
    #dfs = filter_loading(dfs, rolling=10)
    #fig, axes = plot_dfs(dfs)
    #plt.show()
#
    ## 3 - Fitting loading curves
    #dfs_fit, fit_df = fit_loading_dfs(dfs, offset_on=False)
    #fig, axes = plot_dfs_with_fits(dfs_fit)
    #plt.show()
#
    ## 4 - Analysis of the loading curves
    #fig, axes = loading_analysis(fit_df)
    #plt.show()
#
    ## Release, Recapture
    #dfs_rr = make_oscilloscope_df(c.temperature_path)
    #fig, axes = plot_dfs(dfs_rr)
    #plt.show()
#
    ## Fit of Release, Recapture
    #dfs_fit_rr, fit_df_rr = fit_loading_dfs(dfs_rr, offset_on=True)
    #fig, axes = plot_dfs_with_fits(dfs_fit_rr)
    #plt.show()
    #print(fit_df_rr)

    # Loading spectroscopy data
    dfs_spec = make_spectroscopy_df(c.spectroscopy_path)
    #fig, axes = plot_dfs_spectroscopy(dfs_spec, max_column_number=3, plot_PDH_out=False, plot_fit=False)
    #plt.show()

    # Gaussian fits of the transitions (unzoomed)
    dfs_spec = filter_zoomed_spectroscopy(dfs_spec, return_zoomed=False, return_entire=True)
    dfs_spec = calibrate_voltage_to_freq_scale(dfs_spec, 1)
    dfs_spec = mask_dfs(dfs_spec, all_masks=[[(0.10, 0.15)], None, None, None, None])

    dfs_spec, fit_df_spec = fit_spectroscopy_single_dfs(dfs_spec, fct='gaussian',
                                           all_init_params=[(-0.007, 0.12, 0.05, 0.6),
                                                            (-0.007, 0.12, 0.05, 0.6),
                                                            (-0.007, 0.12, 0.05, 0.6),
                                                            (-0.007, 0.12, 0.05, 0.6),
                                                            (-0.007, 0.12, 0.05, 0.6)])

    fig, axes = plot_dfs_spectroscopy(dfs_spec, max_column_number=3, plot_PDH_out=False, plot_fit=True)
    plt.show()
    #fig, axes = plot_dfs_spectroscopy(dfs_spec, max_column_number=2, plot_PDH_out=False)
    # emphasize=[(-0.4, -0.38), (0.65, 0.72)])
    # dfs_fit_spec, fit_df_spec = fit_spectroscopy_dfs(dfs_spec)
    # fig, axes = plot_dfs_spec_with_fits(dfs_fit_spec)
    #plt.show(fig)
    # print(fit_df_spec)

if __name__ == '__main__':
    main()