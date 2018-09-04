def filter_loading(dfs, rolling=1):
    '''
    # 2 - Filtering values that are equal to the minimum value
    '''

    for i, df in enumerate(dfs):
        dfs[i] = df[df != df.min()].rolling(window=rolling, center=True).mean().dropna()
    return dfs
