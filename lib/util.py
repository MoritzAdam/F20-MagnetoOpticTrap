import numpy as np
import pandas as pd
import lib.constants as c
from math import isnan


def get_nearest_in_dataframe(df_series, vals_to_search):
    index = df_series.index.get_loc(vals_to_search, "nearest")
    return df_series.iloc[index]


def get_nearest_index_in_array(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def mask_dfs(dfs, all_masks=None):
    new_dfs = []
    for i, df in enumerate(dfs):
        df, file_name = df
        df_masked = df

        if all_masks is not None:
            if not len(dfs) == len(all_masks):
                raise UserWarning('There is no mask provided for all dataframes')

            masks = all_masks[i]

            if masks is not None:
                for mask in masks:
                    lower, upper = mask
                    lower = get_nearest_in_dataframe(df_masked, lower)
                    upper = get_nearest_in_dataframe(df_masked, upper)
                    lower_ = df_masked.loc[:lower.name]
                    upper_ = df_masked.loc[lower.name:]
                    upper_ = upper_.loc[upper.name:]
                    df_masked = pd.concat([lower_, upper_], axis=0, verify_integrity=False)

        df_masked.columns = ['Masked - Aux in [V]', 'Masked - PDH out [a.u.]']
        df = pd.concat([df, df_masked], axis=1)
        new_dfs.append((df,file_name))
    return new_dfs


def remove_nan_from_masked_column(index, col):
    index_temp = []
    col_temp = []
    for j in range(len(col)):
        if not isnan(col[j]):
            col_temp.append(col[j])
            index_temp.append(index[j])
    return np.asarray(index_temp), np.asarray(col_temp)


def get_multiplet_separation(fit_df, left, right):
    # Use left=1, right=4 for calibration
    separation = fit_df['gauss{}_cen'.format(right)].values[0] - fit_df['gauss{}_cen'.format(left)].values[0]
    error = np.sqrt(fit_df['gauss{}_cen_err'.format(right)].values[0]**2
                    + fit_df['gauss{}_cen_err'.format(left)].values[0]**2)
    return separation, error


def get_definition_zero(fit_df):
    def_zero = fit_df['gauss1_cen'].values[0]
    def_zero_error = fit_df['gauss1_cen_err'].values[0]
    return def_zero, def_zero_error


def get_multiplet_df(df):
    data = {}
    cal, cal_err = get_multiplet_separation(df, 1, 4)
    data['87f287f1'] = (cal, cal_err)
    data['87f285f3'] = get_multiplet_separation(df, 1, 2)
    data['85f285f3'] = get_multiplet_separation(df, 2, 3)
    data['85f287f1'] = get_multiplet_separation(df, 3, 4)

    for key in data:
        val, err = data[key]
        data[key] = (c.RB87_FREQ_SEP_THEORY_F1_F2 * val / cal,
                     (c.RB87_FREQ_SEP_THEORY_F1_F2 * val / cal * np.sqrt((err / val)**2 + (cal_err / cal)**2)))

    data = pd.DataFrame.from_dict(data=data, orient='index',
                                  columns=['separation in frequency [GHz]',
                                           'separation error [GHz]'])
    print('multiplet df', data)
    return data
