from lib.parse_plot_data import get_txt_csv, make_oscilloscope_df, \
    make_spectroscopy_df, plot_dfs, plot_dfs_with_fits, \
    plot_dfs_spectroscopy, plot_dfs_spec_with_fits
from lib.filter_data import filter_loading, filter_zoomed_spectroscopy, \
    calibrate_voltage_to_freq_scale
from lib.fit_data import fit_loading_dfs, fit_spectroscopy_dfs
import matplotlib.pyplot as plt
import lib.constants as c

def main():
    # Loading rates
    # 1 - Starting point
    #dfs = make_oscilloscope_df(c.loading_path)
    #fig, axes = plot_dfs(dfs)
    #plt.show(fig)
#
    ## 2 - Filtering values that are equal to the minimum value
    #dfs = filter_loading(dfs, rolling=10)
    #fig, axes = plot_dfs(dfs)
    #plt.show(fig)
#
    ## 3 - Fitting loading curves
    #dfs_fit, fit_df = fit_loading_dfs(dfs, offset_on=False)
    #fig, axes = plot_dfs_with_fits(dfs_fit)
    #plt.show(fig)
    #print(fit_df)
#
    ## Release, Recapture
    #dfs_rr = make_oscilloscope_df(c.temperature_path)
    #fig, axes = plot_dfs(dfs_rr)
    #plt.show(fig)
#
    ## Fit of Release, Recapture
    #dfs_fit_rr, fit_df_rr = fit_loading_dfs(dfs_rr, offset_on=True)
    #fig, axes = plot_dfs_with_fits(dfs_fit_rr)
    #plt.show(fig)
    #print(fit_df_rr)

    # Loading spectroscopy data
    dfs_spec = make_spectroscopy_df(c.spectroscopy_path)
    fig, axes = plot_dfs_spectroscopy(dfs_spec, max_column_number=3)
    plt.show(fig)

    # Gaussian fits of the transitions (unzoomed)
    dfs_spec = filter_zoomed_spectroscopy(dfs_spec)
    dfs_spec = calibrate_voltage_to_freq_scale(dfs_spec[4:], 100)

    fig, axes = plot_dfs_spectroscopy(dfs_spec, max_column_number=1, plot_PDH_out=False)
                                      #emphasize=[(-0.4, -0.38), (0.65, 0.72)])
    #dfs_fit_spec, fit_df_spec = fit_spectroscopy_dfs(dfs_spec)
    #fig, axes = plot_dfs_spec_with_fits(dfs_fit_spec)
    plt.show(fig)
    #print(fit_df_spec)

if __name__ == '__main__':
    main()