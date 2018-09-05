from lmfit import Parameters, fit_report, minimize
from lmfit.models import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lib.util import remove_nan_from_masked_column

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
        'amp_err': [],
        'offset': [],
        'offset_err': [],
        'tau': [],
        'tau_err': [],
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
        p.add('t0', value=tscan / 5, min=tmin, max=tmax, brute_step=tscan / 20)
        p.add('dt', value=0, min=0, max=tscan, brute_step=tscan / 5)
        if offset_on == False:
            p.add('offset', value=0, vary=offset_on, min=0, max=dv, brute_step=dv / 20)
            p.add('tau', value=1, min=0, max=30, brute_step=5)
            mi = minimize(loading_residual, p, args=(x,), kws={'data': y}, method='powell')
            mi = minimize(loading_residual, mi.params, args=(x,), kws={'data': y}, method='leastsq')
        else:
            p.add('offset', value=0, vary=offset_on, min=0.01, max=dv, brute_step=dv / 20)
            p.add('tau', value=1, min=0, max=60, brute_step=100)
            mi = minimize(loading_residual, p, args=(x,), kws={'data': y}, method='powell')

        dfs[i]['best fit'] = loading_residual(mi.params, x)
        # dfs[i]['init fit'] = loading_residual(mi.init_values, x)

        # storing fit results
        fit_data['amp'].append(mi.params['v1'].value - mi.params['v2'].value)
        if mi.params['v1'].stderr is None:
            fit_data['amp_err'].append(0)
        else:
            fit_data['amp_err'].append(np.sqrt(mi.params['v1'].stderr ** 2 + mi.params['v2'].stderr ** 2))
        fit_data['tau'].append(mi.params['tau'].value)
        fit_data['tau_err'].append(mi.params['tau'].stderr)
        fit_data['offset'].append(mi.params['offset'].value)
        fit_data['offset_err'].append(mi.params['offset'].stderr)
        fit_data['redchi'].append(mi.redchi)

    fit_df = pd.DataFrame(data=fit_data)
    fit_df = fit_df.set_index('file', drop=True).sort_index(level=0)
    return dfs, fit_df


def fit_spectroscopy_entire_dfs(dfs):
    fit_df = []
    return dfs, fit_df


def fit_spectroscopy_single_dfs(dfs, fct='gaussian', all_init_params=None):
    fcts = {
        'gaussian': gaussian,
        'poly_gaussian': poly_gaussian,
        'lorentzian': lorentzian
    }

    fit_data = {
        'file': [file_name for df, file_name in dfs],
        'amp': [],
        'amp_err': [],
        'cen': [],
        'cen_err': [],
        'sig': [],
        'sig_err': [],
        'off': [],
        'off_err': [],
        'redchi': [],
    }

    dfs_fitted = []

    for i, df in enumerate(dfs):
        df, file_name = df
        x_init = df.index.values
        y_init = df.values[:, 0]
        plt.plot(x_init, y_init, 'b')
        x_crop = x_init
        y_crop = df.values[:, 2]

        x_crop, y_crop = remove_nan_from_masked_column(x_crop, y_crop)

        model = Model(fcts[fct], independent_vars=['x'])
        amp, cen, sig, off = all_init_params[i]
        params = model.make_params(amp=amp, cen=cen, sig=sig, off=off)
        fit = model.fit(y_crop, x=x_crop, params=params, method='leastsq', nan_policy='propagate')
        print(fit.fit_report(min_correl=0.25))

        amp, cen, sig, off = fit.best_values.values()
        df['Best fit - Aux in [V]'] = gaussian(x_init, amp=amp, cen=cen, sig=sig, off=off)
        dfs_fitted.append((df, file_name))

        fit_params = fit.params
        fit_data['redchi'].append(fit.redchi)
        for key in fit_data.keys():
            if not key[3:] == '_err' and not key == 'redchi' and not key == 'file':
                fit_data[key].append(fit_params[key].value)
                fit_data[key + '_err'].append(fit_params[key].stderr)

        #plt.plot(x_crop, y_crop, 'r.')
        #plt.plot(x_crop, fit.init_fit, 'g--')
        #plt.plot(x_init, gaussian(x_init, amp=amp, cen=cen, sig=sig, off=off), 'y--')
        #plt.show()

    fit_df = pd.DataFrame(data=fit_data)
    fit_df = fit_df.set_index('file', drop=True).sort_index(level=0)
    print(fit_df)

    return dfs_fitted, fit_df


def gaussian(x, amp, cen, sig, off):
    return amp / (np.sqrt(2 * np.pi) * sig) * np.exp(-((x-cen) / sig)**2 / 2) + off


def lorentzian(x):
    return x


def poly_gaussian(x):
    return x
