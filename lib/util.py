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


def mask_dfs(dfs, all_masks=None, column_to_be_masked='Aux in [V]', split_masks=False):
    new_dfs = []
    for i, df in enumerate(dfs):
        df, file_name = df

        if all_masks is not None:
            if not len(dfs) == len(all_masks):
                raise UserWarning('There is no mask provided for all dataframes')

        if all_masks[i] is not None:
            if split_masks:
                df = _splitted_masks(df, masks=all_masks[i], column_to_be_masked=column_to_be_masked)
            else:
                df = _multi_masks(df, masks=all_masks[i], column_to_be_masked=column_to_be_masked)

        new_dfs.append((df, file_name))
    return new_dfs


def _multi_masks(df, masks, column_to_be_masked='Aux in [V]'):
    if c.MASK_NAME + column_to_be_masked in df.columns.values:
        df = df.drop(c.MASK_NAME + column_to_be_masked, 1, inplace=True)

    df_masked = df.copy()
    for col in df_masked.columns.values:
        if col == column_to_be_masked:
            continue
        else:
            df_masked.drop(col, 1, inplace=True)

    for mask in masks:
        df_masked = _make_excluding_mask(df_masked=df_masked, mask=mask)

    df_masked.columns = ['Masked - ' + column_to_be_masked]
    return pd.concat([df, df_masked], axis=1)


def _splitted_masks(df, masks, column_to_be_masked='Aux in [V]'):
    for i in range(len(masks)):
        if c.MASK_NAME + column_to_be_masked + '-' + str(i) in df.columns.values:
            df = df.drop(c.MASK_NAME + column_to_be_masked + '-' + str(i), 1, inplace=True)

    df_masked = df.copy()
    count = 0

    for col in df_masked.columns.values:
        if col == column_to_be_masked:
            continue
        else:
            df_masked.drop(col, 1, inplace=True)

    for mask in masks:
        df_single_masked = _make_including_mask(df_masked=df_masked, mask=mask)
        df_single_masked.columns = ['Masked - ' + column_to_be_masked + '-' + str(count)]
        df = pd.concat([df, df_single_masked], axis=1)
        count += 1
    return df


def _make_excluding_mask(df_masked, mask):
    lower, upper = mask
    if lower == c.START_TOKEN and upper == c.STOP_TOKEN:
        raise UserWarning('mask can not include complete dataframe at once')

    if lower == c.START_TOKEN:
        lower_ = df_masked.iloc[:0]
    else:
        lower = get_nearest_in_dataframe(df_masked, lower)
        lower_ = df_masked.loc[:lower.name]

    if upper == c.STOP_TOKEN:
        upper_ = df_masked.loc[lower.name:]
        upper_ = upper_.iloc[-1:]
    else:
        upper = get_nearest_in_dataframe(df_masked, upper)
        upper_ = df_masked.iloc[0:]
        upper_ = upper_.loc[upper.name:]

    df_masked = pd.concat([lower_, upper_], axis=0, verify_integrity=False)
    return df_masked


def _make_including_mask(df_masked, mask):
    lower, upper = mask
    lower = get_nearest_in_dataframe(df_masked, lower)
    upper = get_nearest_in_dataframe(df_masked, upper)
    return df_masked.loc[lower.name:upper.name]


def _remove_nan_from_masked_column(index, col):
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
    error = np.sqrt(fit_df['gauss{}_sig'.format(right)].values[0] ** 2
                    + fit_df['gauss{}_sig'.format(left)].values[0] ** 2)
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
                     (c.RB87_FREQ_SEP_THEORY_F1_F2 * val / cal * np.sqrt((err / val) ** 2 + (cal_err / cal) ** 2)))

    data = pd.DataFrame.from_dict(data=data, orient='index',
                                  columns=['separation in frequency [GHz]',
                                           'separation error [GHz]'])
    print('multiplet df', data)
    return data
