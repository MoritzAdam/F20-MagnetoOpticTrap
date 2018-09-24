import lib.constants as c
from pandas import Series
from lib.parse_plot_data import make_spectroscopy_df
import numpy as np


def filter_loading(dfs, rolling=1):
    # Filtering values that are equal to the minimum value
    for i, df in enumerate(dfs):
        dfs[i] = df[df != df.min()].rolling(window=rolling, center=True).mean().dropna()
    return dfs


def filter_zoomed_spectroscopy(dfs, return_zoomed=False, return_entire=True):
    '''
    :param dfs: list of pandas dataframe objects
    :param return_zoomed: bool; if false all 'zoomed' dfs will be filtered, if true only zoomed dfs will be returned
    :param return_entire: bool; if true, df for entire spectrum will be returned
    :return: list of pandas dataframe objects
    '''
    filtered_dfs = []
    for i, df in enumerate(dfs):
        df, file_name = df
        # df.index,df.values[:,1]
        if not return_zoomed:
            if not file_name[5:] == 'zoom' and 'all' not in file_name:
                filtered_dfs.append((df, file_name))
        else:
            if file_name[5:] == 'zoom':
                filtered_dfs.append((df, file_name))
        if return_entire:
            if 'all' in file_name:
                filtered_dfs.append((df, file_name))
    return filtered_dfs


def calibrate_voltage_to_freq_scale(dfs, calibration_factor, definition_zero):
    '''
    calibrates voltage axis with theoretical freq difference for the two 87Rb fine structure transitions
    changes df index from voltage to calibrated freq, drops voltage values
    :param dfs: list of pd.Dataframe objects
    :param calibration_factor: double; fraction delta freq/delta voltage [THz/V]
    :return: list of pd.Dataframe objects
    '''
    calibrated_dfs = []

    for df in dfs:
        if not len(df) == 1:
            df, file_name = df

        calibrated_x = c.RB87_FREQ_SEP_THEORY_F1_F2 * (df.index - definition_zero) / calibration_factor
        df['Frequency [GHz]'] = Series(calibrated_x, index=df.index)
        df = df.reset_index(drop=True)
        df = df.set_index('Frequency [GHz]')

        if not len(df) == 1:
            calibrated_dfs.append((df, file_name))
        else:
            calibrated_dfs.append(df)

    return calibrated_dfs


def subtract_gaussian_fit(dfs):
    filtered_dfs = []
    for df in dfs:
        if not len(df) == 1:
            df, file_name = df

        df['Aux in minus Best fit [V]'] = Series(data=df['Aux in [V]'].values - df['Best fit - Aux in [V]'].values,
                                                 index=df.index)

        if not len(df) == 1:
            filtered_dfs.append((df, file_name))
        else:
            filtered_dfs.append(df)

    return dfs


def filter_recapture(dfs):
    upperlimits = [None, None, 0.006, 0.008, None, None, 0.016, 0.016, 0.016, None, None, None, None, None, 0.03, 0.03,
                   0.03, 0.03, 0.03, 0.035, None, None, None, None, None, 0.06, 0.08, 0.12, 0.13, 0.14, 0.14]
    for i, df in enumerate(dfs):
        if upperlimits[i] is not None:
            limit = df.index.get_loc(upperlimits[i], method='nearest')
            dfs[i] = df.iloc[:limit]
    return dfs


if __name__ == '__main__':
    dfs_spec = make_spectroscopy_df(c.spectroscopy_path)
    dfs_spec = filter_zoomed_spectroscopy(dfs_spec)
