import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import lib.constants as c
from math import ceil
from lib.util import get_nearest_in_dataframe, _remove_nan_from_masked_column

def import_dict(str):
    det = {}
    with open('../data/'+str) as f:
        for line in f:
            (key, val) = line.split()
            det[key] = float(val)
    return det

def get_txt_csv(path):
    files = os.listdir(path)
    file_names = [file for file in files if file.endswith(('.txt', '.TXT', '.CSV', '.csv'))]
    file_paths = [os.path.join(path, file) for file in file_names]
    return file_names, file_paths


def make_oscilloscope_df(path):
    file_names, file_paths = get_txt_csv(path)
    dfs = []
    for file_path, file_name in zip(file_paths, file_names):
        df = pd.read_csv(file_path, header=None)
        df.drop([0, 1, 2, 5], axis=1, inplace=True)
        #         df = df.astype(np.float64)
        #         df.index = df.index.astype(np.float64)
        df.set_index(3, inplace=True)
        df.index.name = 'time [s]'
        df.columns = [file_name[:-4]]
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    return dfs


def make_spectroscopy_df(path):
    file_names, file_paths = get_txt_csv(path)
    dfs = []
    for file_path, file_name in zip(file_paths, file_names):
        df = pd.read_table(file_path, header=0, sep='\t', index_col=0, dtype=np.float64)
        dfs.append((df, file_name[6:-4]))
    # df = pd.concat(dfs, axis=1)
    return dfs


def plot_dfs(dfs, style, recapture=None):
    plot_columns = ceil(len(dfs) / 4)
    fig, axes = plt.subplots(plot_columns, 4, figsize=(15, plot_columns * 2), facecolor='w', edgecolor='k')
    dict = import_dict(style[1])
    axes = axes.ravel()
    for i, df in enumerate(dfs):
        axes[i].plot(df.index, df.iloc[:, 0].values, '.', markersize=c.PLOT_MARKERSIZE, color=c.BLUE)
        #         axes[i].plot(df.index,df['init fit'].values, 'k--')
        if df.get('best fit') is not None:
            axes[i].plot(df.index, df['best fit'].values, 'r-', markersize=c.PLOT_MARKERSIZE, color=c.RED)
        if recapture is not None:
            axes[i].plot(df.index, recapture[0][i]*np.ones(len(df.index)), '--', color=c.GREEN)
            axes[i].plot(df.index, recapture[1][i]*np.ones(len(df.index)), '--', color=c.GREEN)
        axes[i].set_title(style[0]+str(dict[df.columns[0]])+style[2], fontweight='semibold', size='10')
    for i in range(0, plot_columns):
        axes[i*4].set_ylabel(style[3], style='italic')
    for i in range(4*plot_columns-4, 4*plot_columns):
        axes[i].set_xlabel(style[4], style='italic')
    for i in range(len(dfs),4*plot_columns):
        axes[i].set_visible(False)
    plt.tight_layout()


# TODO: label transitions in plots
# TODO: split up functions in smaller parts
def plot_dfs_spectroscopy(dfs, max_column_number, x_label, y_label, fit_data=None, plot_initial=True, plot_PDH_out=False, plot_fit=False,
                          plot_deriv=False, plot_data_with_subtracted_fit=False, plot_hyperfine_fit=False,
                          use_global_zoom_for_hyperfine=False, subplot_title_addition='', emphasize=None,
                          use_splitted_masks=False, use_automated_fit_plot_barrier=False, column_name='Aux in [V]', masks=None):
    if not use_global_zoom_for_hyperfine:
        zoom = [(0, -1), (0, -1), (0, -1), (0, -1)]
    else:
        zoom = c.GLOBAL_ZOOM_FOR_HYPERFINE

    plot_columns = ceil(len(dfs)/max_column_number)
    fig, axis = plt.subplots(plot_columns, max_column_number,
                             figsize=(15, plot_columns*2), facecolor='w', edgecolor='k')

    if not max_column_number == 1:
        axis = axis.ravel()
    else:
        axis = np.array([axis])

    for i, df in enumerate(dfs):
        df, file_name = df
        add_axis = axis[i].twinx()

        if plot_initial:
            axis[i].plot(df.index, df.values[:, 0], '.', color='steelblue', markersize=c.PLOT_MARKERSIZE)

        if emphasize is not None:
            for single_emphasize in emphasize:
                lower, upper = single_emphasize
                lower = get_nearest_in_dataframe(df, lower)
                upper = get_nearest_in_dataframe(df, upper)
                df_to_be_emphasized = df.loc[lower.name:upper.name]
                axis[i].plot(df_to_be_emphasized.index, df_to_be_emphasized.values[:, 'Aux in [V]'], '.', color='orangered',
                             markersize=c.PLOT_MARKERSIZE)

        if plot_PDH_out:
            add_axis.plot(df.index, df.loc[:, 'PDH out [a.u.]'].values, '.', color='grey', alpha=0.3, markersize=c.PLOT_MARKERSIZE)

        if plot_data_with_subtracted_fit:
            add_axis.plot(df.index[zoom[i][0]:zoom[i][1]],
                          df.loc[:, 'Aux in minus Best fit [V]']
                          .values[zoom[i][0]:zoom[i][1]],
                          '.', color='limegreen', markersize=c.PLOT_MARKERSIZE)

        if plot_fit:
            if use_splitted_masks:
                count = 0
                mask = masks[i]
                for single_mask in mask:
                    column_extension = '-' + str(count)
                    add_axis.plot(df.index[zoom[i][0]:zoom[i][1]],
                                  df.loc[:, c.MASK_NAME + column_name + column_extension]
                                  .values[zoom[i][0]:zoom[i][1]],
                                  '.', color='black', markersize=c.PLOT_MARKERSIZE)

                    if not use_automated_fit_plot_barrier:
                        lower_fit_plot_barrier = zoom[i][0]
                        upper_fit_plot_barrier = zoom[i][1]

                        add_axis.plot(df.index[lower_fit_plot_barrier:upper_fit_plot_barrier].values,
                                      df.loc[:, 'Best fit - ' + column_name + column_extension]
                                      .values[lower_fit_plot_barrier:upper_fit_plot_barrier],
                                      '-', color='r', alpha=0.7, markersize=c.PLOT_MARKERSIZE)

                    else:
                        if fit_data is None:
                            raise UserWarning('fit_data has to be provided if use_automated_fit_plot_barrier is True')

                        cen = fit_data.loc[file_name + column_extension, 'cen']

                        try:
                            gamma = fit_data.loc[file_name + column_extension, 'gamma']
                        except:
                            try:
                                gamma = fit_data.loc[file_name + column_extension, 'sig']
                            except:
                                raise UserWarning('use_automated_fit_plot_barrier can not find key')

                        lower_fit_plot_barrier = get_nearest_in_dataframe(df, cen - 7 * gamma).name
                        upper_fit_plot_barrier = get_nearest_in_dataframe(df, cen + 7 * gamma).name

                        series = df.loc[lower_fit_plot_barrier:upper_fit_plot_barrier, 'Best fit - ' + column_name + column_extension]

                        add_axis.plot(series.index.values, series.values,
                                      '-', color='r', alpha=0.7, markersize=c.PLOT_MARKERSIZE)

                    count += 1

            else:
                if plot_data_with_subtracted_fit:
                    ax = add_axis
                else:
                    ax = axis[i]
                ax.plot(df.index[zoom[i][0]:zoom[i][1]], df.loc[:, c.MASK_NAME + column_name].values[zoom[i][0]:zoom[i][1]], '.', color='black',
                             markersize=c.PLOT_MARKERSIZE)
                ax.plot(df.index[zoom[i][0]:zoom[i][1]], df.loc[:, 'Best fit - ' + column_name].values[zoom[i][0]:zoom[i][1]], '-', color='r', alpha=0.7,
                             markersize=c.PLOT_MARKERSIZE)



        if plot_hyperfine_fit:
            if not plot_data_with_subtracted_fit:
                raise UserWarning('plot_data_with_subtracted_fit has to be True when plot_hyperfine_fit is True')
            add_axis.plot(df.index[zoom[i][0]:zoom[i][1]],
                          df.loc[:, 'Masked - Aux in minus Best fit [V]']
                          .values[zoom[i][0]:zoom[i][1]],
                          '.', color='black', markersize=c.PLOT_MARKERSIZE)

        if plot_deriv:
            add_axis = axis[i].twinx()
            add_axis.plot(df.index[zoom[i][0] + 1:zoom[i][1]],
                          np.diff(np.asarray(df.loc[:, 'Aux in [V]'].values))[zoom[i][0]:zoom[i][1]],
                          '-', color='limegreen', alpha=0.4, markersize=c.PLOT_MARKERSIZE)

        axis[i].set_title(txt_title(file_name) + subplot_title_addition, fontsize=10)
        axis[i].set_xlabel(x_label)
        axis[i].set_ylabel(y_label)


    if not max_column_number == 1:
        plt.tight_layout()

    return fig, axis


def txt_title(file_name):
    if not file_name[:3] == 'all':
        return r'$^{%s}Rb; \ F=%s \rightarrow F´$'%(file_name[0:2], file_name[3:4])
    else:
        return r'$^{85}Rb \ and \ ^{87}Rb \ spectrum$'

