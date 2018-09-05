def get_nearest(df_series, vals_to_search):
    index = df_series.index.get_loc(vals_to_search, "nearest")
    return df_series.iloc[index]