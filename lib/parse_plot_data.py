import matplotlib.pyplot as plt
import pandas as pd
import os
from math import ceil


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
        axis[i].plot(df.index, df.values, '.', markersize=2)
        axis[i].set_title(df.columns[0])
    return fig, axis


def plot_dfs_with_fits(dfs):
    plot_columns = ceil(len(dfs) / 4)
    fig, axes = plt.subplots(plot_columns, 4, figsize=(15, plot_columns * 2), facecolor='w', edgecolor='k')
    axes = axes.ravel()
    for i, df in enumerate(dfs):
        axes[i].plot(df.index, df.iloc[:, 0].values, '.')
        #         axes[i].plot(df.index,df['init fit'].values, 'k--')
        axes[i].plot(df.index, df['best fit'].values, 'r-')
    return fig, axes


def plot_dfs_spectroscopy(dfs):
    plot_columns = ceil(len(dfs)/3)
    fig, axis = plt.subplots(plot_columns, 3, figsize=(15, plot_columns*2), facecolor='w', edgecolor='k')
    plt.tight_layout()
    axis = axis.ravel()
    for i, df in enumerate(dfs):
        df, file_name = df
        axis[i].plot(df.index,df.values[:,0], '.', markersize=2)
        add_axis = axis[i].twinx()
        add_axis.plot(df.index,df.values[:,1], '.', color='grey', alpha=0.3, markersize=2)
        axis[i].set_title(txt_title(file_name), fontsize=8)
    return fig, axis


def txt_title(file_name):
    return file_name
