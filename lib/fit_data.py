import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lib.util import remove_nan_from_masked_column, get_nearest_index_in_array
from lmfit import Parameters, minimize
from lmfit.models import Model, LinearModel, ExponentialModel


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
        't0': [],
        'dt': [],
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
        fit_data['t0'].append(mi.params['t0'].value)
        fit_data['dt'].append(mi.params['dt'].value)
        fit_data['redchi'].append(mi.redchi)

    fit_df = pd.DataFrame(data=fit_data)
    fit_df = fit_df.set_index('file', drop=True).sort_index(level=0)
    return dfs, fit_df


def _initialize_fit_data_df(fct, dfs):
    fit_params = {}
    if fct == 'gaussian':
        fit_params.update({
            'file': [file_name for df, file_name in dfs],
            'amp': [],
            'amp_err': [],
            'cen': [],
            'cen_err': [],
            'sig': [],
            'sig_err': [],
            'off': [],
            'off_err': [],
            'redchi': []
        })

    if fct == 'lorentzian':
        fit_params.update({
            'file': [file_name for df, file_name in dfs],
            'cen': [],
            'cen_err': [],
            'gamma': [],
            'gamma_err': [],
            'off': [],
            'off_err': [],
            'redchi': []
        })

    if fct == 'double_gaussian' or fct == 'poly_gaussian':
        if fct == 'double_gaussian':
            gauss_count = 2
            lin_count = 1
        else:
            gauss_count = 4
            lin_count = 2
        for i in range(1, gauss_count + 1):
            fit_params['gauss{}_amp'.format(i)] = []
            fit_params['gauss{}_cen'.format(i)] = []
            fit_params['gauss{}_sig'.format(i)] = []
            fit_params['gauss{}_off'.format(i)] = []
        for i in range(1, lin_count + 1):
            fit_params['linear{}_intercept'.format(i)] = []
            fit_params['linear{}_slope'.format(i)] = []
        keys = list(fit_params.keys())
        for key in keys:
            fit_params[key + '_err'] = []
        fit_params['file'] = [file_name for df, file_name in dfs],
        fit_params['redchi'] = []

    return fit_params


def fit_spectroscopy_dfs(dfs, fct='gaussian', all_init_params=None, column_to_fit='Aux in [V]'):
    fcts = {
        'gaussian': gaussian,
        'double_gaussian': double_gaussian,
        'poly_gaussian': poly_gaussian,
        'lorentzian': lorentzian
    }

    if fct not in fcts.keys():
        raise UserWarning('unknown fit function')

    fit_stat = _initialize_fit_data_df(fct, dfs)
    dfs_fitted = []

    for i, df in enumerate(dfs):
        df, file_name = df
        x_crop = df.index.values
        y_crop = df.loc[:, 'Masked - ' + column_to_fit].values

        x_crop, y_crop = remove_nan_from_masked_column(x_crop, y_crop)

        model = _make_model(fcts[fct])
        params = _get_init_params(fct, all_init_params, model, x_crop, y_crop, i)

        fit = model.fit(y_crop, x=x_crop, params=params, method='leastsq', nan_policy='propagate')
        print(fit.fit_report(min_correl=0.25))

        fit_stat = _save_fit_params(fit, fit_stat)

        df = _save_fit_in_df(df=df, fit=fit, column_to_fit=column_to_fit)
        dfs_fitted.append((df, file_name))

    fit_df = pd.DataFrame(data=fit_stat)
    fit_df = fit_df.set_index('file', drop=True).sort_index(level=0)

    return dfs_fitted, fit_df


def _get_init_params(fct, all_init_params, model, x_crop, y_crop, i):
    if fct == 'gaussian':
        if all_init_params and all_init_params[i] is not None:
            amp, cen, sig, off = all_init_params[i]
        else:
            # guess initial params
            amp = np.max(y_crop)
            cen = x_crop[np.argmax(y_crop)]
            off = np.max(x_crop)
            sig = abs(cen - x_crop[get_nearest_index_in_array(y_crop, (amp - off) / 2)])
        params = model.make_params(amp=amp, cen=cen, sig=sig, off=off)

    if fct == 'lorentzian':
        if all_init_params and all_init_params[i] is not None:
            cen, gamma, off = all_init_params[i]
        else:
            # guess initial params
            raise UserWarning('please provide initial params; for poly_gaussian params guess is not yet implemented')
        params = model.make_params(cen=cen, gamma=gamma, off=off)

    if fct == 'double_gaussian':
        if all_init_params and all_init_params[i] is not None:
            gauss1_amp, gauss2_amp, gauss1_cen, gauss2_cen, \
            gauss1_sig, gauss2_sig, gauss1_off, gauss2_off, \
            linear1_intercept, linear1_slope = all_init_params[i]
        else:
            # guess initial params
            raise UserWarning('please provide initial params; for poly_gaussian params guess is not yet implemented')
        params = model.make_params(gauss1_amp=gauss1_amp, gauss2_amp=gauss2_amp,
                                   gauss1_cen=gauss1_cen, gauss2_cen=gauss2_cen,
                                   gauss1_sig=gauss1_sig, gauss2_sig=gauss2_sig,
                                   gauss1_off=gauss1_off, gauss2_off=gauss2_off,
                                   linear1_intercept=linear1_intercept, linear1_slope=linear1_slope)

    if fct == 'poly_gaussian':
        if all_init_params and all_init_params[i] is not None:
            gauss1_amp, gauss2_amp, gauss3_amp, gauss4_amp, \
            gauss1_cen, gauss2_cen, gauss3_cen, gauss4_cen, \
            gauss1_sig, gauss2_sig, gauss3_sig, gauss4_sig, \
            gauss1_off, gauss2_off, gauss3_off, gauss4_off,\
            linear1_intercept, linear2_intercept,\
            linear1_slope, linear2_slope = all_init_params[i]
        else:
            # guess initial params
            raise UserWarning('please provide initial params; for poly_gaussian params guess is not yet implemented')
        params = model.make_params(gauss1_amp=gauss1_amp, gauss2_amp=gauss2_amp, gauss3_amp=gauss3_amp, gauss4_amp=gauss4_amp,
                                   gauss1_cen=gauss1_cen, gauss2_cen=gauss2_cen, gauss3_cen=gauss3_cen, gauss4_cen=gauss4_cen,
                                   gauss1_sig=gauss1_sig, gauss2_sig=gauss2_sig, gauss3_sig=gauss3_sig, gauss4_sig=gauss4_sig,
                                   gauss1_off=gauss1_off, gauss2_off=gauss2_off, gauss3_off=gauss3_off, gauss4_off=gauss4_off,
                                   linear1_intercept=linear1_intercept, linear2_intercept=linear2_intercept,
                                   linear1_slope=linear1_slope, linear2_slope=linear2_slope)
    return params


def _make_model(fct):
    if not fct == poly_gaussian and not fct == double_gaussian:
        return Model(fct, independent_vars=['x'])
    else:
        return fct()


def gaussian(x, amp, cen, sig, off):
    return amp / (np.sqrt(2 * np.pi) * sig) * np.exp(-((x-cen) / sig)**2 / 2) + off


def double_gaussian():
    gauss1 = Model(gaussian, independent_vars=['x'], prefix='gauss1_')
    gauss2 = Model(gaussian, independent_vars=['x'], prefix='gauss2_')
    linear1 = LinearModel(independent_vars=['x'], prefix='linear1_')
    model = gauss1 + linear1 + gauss2
    return model


def poly_gaussian():
    gauss1 = Model(gaussian, independent_vars=['x'], prefix='gauss1_')
    gauss2 = Model(gaussian, independent_vars=['x'], prefix='gauss2_')
    gauss3 = Model(gaussian, independent_vars=['x'], prefix='gauss3_')
    gauss4 = Model(gaussian, independent_vars=['x'], prefix='gauss4_')
    linear1 = LinearModel(independent_vars=['x'], prefix='linear1_')
    linear2 = LinearModel(independent_vars=['x'], prefix='linear2_')
    model = gauss1 + gauss2 + linear1 + gauss3 + linear2 + gauss4
    return model


def lorentzian(x, cen, gamma, off):
    return 1 / (np.pi * gamma) * 1 / (1 + (x-cen)**2 / gamma**2) + off


def _save_fit_in_df(df, fit, column_to_fit):
    x_init = df.index.values
    df['Best fit - ' + column_to_fit] = fit.eval(x=x_init)
    return df


def _save_fit_params(fit, fit_data):
    fit_params = fit.params
    fit_data['redchi'].append(fit.redchi)
    for key in fit_data.keys():
        if not key[-4:] == '_err' and not key == 'redchi' and not key == 'file':
            fit_data[key].append(fit_params[key].value)
            fit_data[key + '_err'].append(fit_params[key].stderr)
    return fit_data
