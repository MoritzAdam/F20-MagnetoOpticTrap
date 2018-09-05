from lib.parse_plot_data import make_spectroscopy_df
import lib.constants as c
from pandas import Series
import numpy as np


def filter_loading(dfs, rolling=1):
    '''
    # 2 - Filtering values that are equal to the minimum value
    '''

    for i, df in enumerate(dfs):
        dfs[i] = df[df != df.min()].rolling(window=rolling, center=True).mean().dropna()
    return dfs


def filter_zoomed_spectroscopy(dfs):
    filtered_dfs = []
    for i, df in enumerate(dfs):
        df, file_name = df
        # df.index,df.values[:,1]
        if not file_name[5:] == 'zoom':
            filtered_dfs.append((df, file_name))
    return filtered_dfs


def calibrate_voltage_to_freq_scale(dfs, calibration_factor):
    '''
    calibrates voltage axis with theoretical freq difference for the two 87Rb fine structure transitions
    changes df index from voltage to calibrated freq, drops voltage values
    :param dfs: list of pd.Dataframe objects
    :param calibration_factor: double; fraction delta freq/delta voltage [THz/V]
    :return: list of pd.Dataframe objects
    '''
    calibrated_dfs = []

    for df in dfs:
        if len(df) == 1:
            calibrated_x = df.index * calibration_factor #np.random.randn(len(df.index))
            df['Frequency [THz]'] = Series(calibrated_x, index=df.index)
            df = df.reset_index(drop=True)
            df = df.set_index('Frequency [THz]')
            calibrated_dfs.append(df)
        else:
            df, file_name = df
            calibrated_x = df.index * calibration_factor  # np.random.randn(len(df.index))
            df['Frequency [THz]'] = Series(calibrated_x, index=df.index)
            df = df.reset_index(drop=True)
            df = df.set_index('Frequency [THz]')
            calibrated_dfs.append((df, file_name))

    return calibrated_dfs


if __name__ == '__main__':
    dfs_spec = make_spectroscopy_df(c.spectroscopy_path)
    dfs_spec = filter_zoomed_spectroscopy(dfs_spec)
