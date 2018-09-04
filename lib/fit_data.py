from lmfit import Parameters, fit_report, minimize
import pandas as pd
import numpy as np


def loading_residual(pars, x, data=None):
    vals = pars.valuesdict()
    v1 = vals['v1']
    v2 = vals['v2']
    offset = vals['offset']
    t0 = vals['t0']
    dt = vals['dt']
    tau = vals['tau']

    model = v2 + (v1 - v2) * (1 - np.exp(-np.abs((x - t0 - dt) / tau))) + offset
    model[x <= t0 + dt] = v2
    model[x <= t0] = v1
    if data is None:
        return model
    return model - data


def fit_loading_dfs(dfs, offset_on=False):
    fit_data = {
        'file': [df.columns[0] for df in dfs],
        'amp': [],
        'offset': [],
        #         'amp_std': [],
        'tau': [],
        #         'tau_std': [],
        'redchi': [],
    }
    for i, df in enumerate(dfs):
        x = df.index.values
        y = df.iloc[:, 0].values

        # predicting parameters
        vmin = np.min(y)
        vmax = np.max(y)
        dv = vmax - vmin
        tmin = np.min(x)
        tmax = np.max(x)
        tscan = tmax - tmin

        p = Parameters()
        p.add('v1', value=vmax, min=vmax - dv / 10, max=vmax)
        p.add('v2', value=vmin, min=vmin, max=np.min(y) + 0.03)
        p.add('offset', value=0, vary=offset_on, min=0, max=dv, brute_step=dv / 20)
        p.add('t0', value=tscan / 5, min=tmin, max=tmax, brute_step=tscan / 20)
        p.add('dt', value=0, min=0, max=tscan, brute_step=tscan / 5)
        p.add('tau', value=1, min=0, max=60, brute_step=10)

        mi = minimize(loading_residual, p, args=(x,), kws={'data': y}, method='powell')
        #         mi = minimize(loading_residual, mi.params, args=(x,), kws={'data': y}, method='leastsq')
        dfs[i]['best fit'] = loading_residual(mi.params, x)
        #         dfs[i]['init fit'] = loading_residual(mi.init_values, x)

        # storing fit results
        fit_data['amp'].append(mi.params['v1'].value - mi.params['v2'].value)
        fit_data['tau'].append(mi.params['tau'].value)
        fit_data['offset'].append(mi.params['offset'].value)
        fit_data['redchi'].append(mi.redchi)

    fit_df = pd.DataFrame(data=fit_data)
    fit_df = fit_df.set_index('file', drop=True).sort_index(level=0)
    return dfs, fit_df
