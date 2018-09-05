import matplotlib.pyplot as plt
import pandas as pd
import os
from math import ceil
import lib.constants as c
from lib.util import get_nearest


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
        df = pd.read_table(file_path, header=0, sep='\t', index_col=0)
        dfs.append((df, file_name[6:-4]))
    # df = pd.concat(dfs, axis=1)
    return dfs


def plot_dfs(dfs):
    plot_columns = ceil(len(dfs) / 4)
    fig, axis = plt.subplots(plot_columns, 4, figsize=(15, plot_columns * 2), facecolor='w', edgecolor='k')

    axis = axis.ravel()
    for i, df in enumerate(dfs):
        axis[i].plot(df.index, df.values, '.', markersize=c.PLOT_MARKERSIZE)
        axis[i].set_title(df.columns[0])
    return fig, axis


def plot_dfs_with_fits(dfs):
    plot_columns = ceil(len(dfs) / 4)
    fig, axes = plt.subplots(plot_columns, 4, figsize=(15, plot_columns * 2), facecolor='w', edgecolor='k')
    axes = axes.ravel()
    for i, df in enumerate(dfs):
        axes[i].plot(df.index, df.iloc[:, 0].values, '.', markersize=c.PLOT_MARKERSIZE)
        #         axes[i].plot(df.index,df['init fit'].values, 'k--')
        axes[i].plot(df.index, df['best fit'].values, 'r-', markersize=c.PLOT_MARKERSIZE)
    return fig, axes


def plot_dfs_spectroscopy(dfs, max_column_number, plot_PDH_out = True,
                          subplot_title_addition='', global_title='', emphasize=None):
    plot_columns = ceil(len(dfs)/max_column_number)
    fig, axis = plt.subplots(plot_columns, max_column_number,
                             figsize=(15, plot_columns*2), facecolor='w', edgecolor='k')

    if not max_column_number == 1:
        axis = axis.ravel()
        for i, df in enumerate(dfs):
            df, file_name = df
            axis[i].plot(df.index, df.values[:, 0], '.', markersize=c.PLOT_MARKERSIZE)
            if emphasize is not None:
                for single_emphasize in emphasize:
                    lower, upper = single_emphasize
                    lower = get_nearest(df, lower)
                    upper = get_nearest(df, upper)
                    df_to_be_emphasized = df.loc[lower.name:upper.name]
                    axis[i].plot(df_to_be_emphasized.index, df_to_be_emphasized.values[:, 0], '.', color='r',
                                 markersize=c.PLOT_MARKERSIZE)
            if plot_PDH_out:
                add_axis = axis[i].twinx()
                add_axis.plot(df.index,df.values[:, 1], '.', color='grey', alpha=0.3, markersize=c.PLOT_MARKERSIZE)
            axis[i].set_title(txt_title(file_name) + subplot_title_addition, fontsize=10)
        plt.tight_layout()

    else:
        df, file_name = dfs[0]
        axis.plot(df.index, df.values[:, 0], '.', markersize=c.PLOT_MARKERSIZE)
        if emphasize is not None:
            for single_emphasize in emphasize:
                lower, upper = single_emphasize
                lower = get_nearest(df, lower)
                upper = get_nearest(df, upper)
                df_to_be_emphasized = df.loc[lower.name:upper.name]
                axis.plot(df_to_be_emphasized.index, df_to_be_emphasized.values[:, 0], '.', color='r',
                          markersize=c.PLOT_MARKERSIZE)
        if plot_PDH_out:
            add_axis = axis.twinx()
            add_axis.plot(df.index, df.values[:, 1], '.', color='grey', alpha=0.3, markersize=c.PLOT_MARKERSIZE)
        axis.set_title(txt_title(file_name) + subplot_title_addition, fontsize=10)

    return fig, axis


def plot_dfs_spec_with_fits(dfs, max_column_number):
    return dfs


def txt_title(file_name):
    if not file_name[:3] == 'all':
        return r'$^{%s}Rb; \ F=%s \rightarrow F´$'%(file_name[0:2], file_name[3:4])
    else:
        return r'$^{85}Rb \ and \ ^{87}Rb \ spectrum$'

