import numpy as np
import pandas as pd
from math import isnan

def get_nearest(df_series, vals_to_search):
    index = df_series.index.get_loc(vals_to_search, "nearest")
    return df_series.iloc[index]


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
                    lower = get_nearest(df_masked, lower)
                    upper = get_nearest(df_masked, upper)
                    lower_ = df_masked.loc[:lower.name]
                    upper_ = df_masked.loc[lower.name:]
                    upper_ = upper_.loc[upper.name:]
                    df_masked = pd.concat([lower_, upper_], axis=0, verify_integrity=True)

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
