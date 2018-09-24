import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lib.util import _remove_nan_from_masked_column, get_nearest_index_in_array
from lmfit import Parameters, minimize
from lmfit.models import Model, LinearModel, ExponentialModel, LorentzianModel


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


def _initialize_fit_data_df(fct, dfs, use_splitted_masks=False, masks=None, number_unique_lorentzians=None):
    fit_params = {}
    if fct == 'gaussian':
        fit_params.update({
            'amp': [],
            'cen': [],
            'sig': [],
            'off': []
        })

    if fct == 'lorentzian':
        fit_params.update({
            'amp': [],
            'cen': [],
            'gamma': [],
            'off': []
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

    if fct == 'poly_lorentzian':
        if number_unique_lorentzians is None:
            raise UserWarning('number of unique lorentzians need to be provided for the fit')
        for i in range(1, number_unique_lorentzians + 1):
            fit_params['lorentzian{}_amp'.format(i)] = []
            fit_params['lorentzian{}_cen'.format(i)] = []
            fit_params['lorentzian{}_gamma'.format(i)] = []
        fit_params['linear1_intercept'] = []
        fit_params['linear1_slope'] = []

    for key in list(fit_params.keys()):
        fit_params[key + '_err'] = []
    fit_params['file'] = [file_name for df, file_name in dfs]
    fit_params['redchi'] = []

    if use_splitted_masks:
        new_files = []
        for j, mask in enumerate(masks):
            for i, single_mask in enumerate(mask):
                new_files.append(fit_params['file'][j] + '-{}'.format(i))
        fit_params['file'] = new_files
    return fit_params


def fit_spectroscopy_dfs(dfs, fct='gaussian', init_params=None, column_to_fit='Aux in [V]',
                         use_splitted_masks=False, masks=None, use_multiple_lorentzians=False):
    fcts = {
        'gaussian': gaussian,
        'double_gaussian': double_gaussian,
        'poly_gaussian': poly_gaussian,
        'lorentzian': lorentzian,
        'poly_lorentzian': poly_lorentzian
    }

    if fct not in fcts.keys():
        raise UserWarning('unknown fit function')

    if use_multiple_lorentzians:
        unique_lorentzians = np.asarray([n.pop('number_unique_lorentzians', None) for n in init_params])
        max_numb_unique_lorentzians = max(unique_lorentzians)
    else:
        unique_lorentzians = [None, None, None, None]
        max_numb_unique_lorentzians = None
    fit_stat = _initialize_fit_data_df(fct, dfs, use_splitted_masks=use_splitted_masks, masks=masks,
                                       number_unique_lorentzians=max_numb_unique_lorentzians)
    dfs_fitted = []

    for i, df in enumerate(dfs):
        df, file_name = df
        x_crop = df.index.values

        if init_params is not None:
            single_init_params = init_params[i]
        else:
            single_init_params = init_params

        if use_splitted_masks:
            mask = masks[i]
            count = 0
            for single_mask in mask:
                column_extension = '-' + str(count)
                y_crop = df.loc[:, 'Masked - ' + column_to_fit + column_extension].values
                df, fit_stat = fit_single_spectroscopy_column(df, x_crop, y_crop, single_init_params, i, fcts, fct,
                                                              fit_stat, column_to_fit,
                                                              column_extension=column_extension,
                                                              number_unique_lorentzians=unique_lorentzians[i])
                count += 1
        else:
            y_crop = df.loc[:, 'Masked - ' + column_to_fit].values
            df, fit_stat = fit_single_spectroscopy_column(df, x_crop, y_crop, single_init_params, i, fcts, fct,
                                                          fit_stat,
                                                          column_to_fit,
                                                          number_unique_lorentzians=unique_lorentzians[i])
        dfs_fitted.append((df, file_name))

    fit_df = pd.DataFrame(data=fit_stat)
    fit_df = fit_df.set_index('file', drop=True).sort_index(level=0)

    return dfs_fitted, fit_df


def fit_single_spectroscopy_column(df, x_crop, y_crop, init_params, i, fcts, fct, fit_stat, column_to_fit,
                                   column_extension='', number_unique_lorentzians=None):
    x_crop, y_crop = _remove_nan_from_masked_column(x_crop, y_crop)

    model = _make_model(fcts[fct], number_unique_lorentzians=number_unique_lorentzians)
    params = _get_init_params(fct, init_params, model, x_crop, y_crop, i)

    fit = model.fit(y_crop, x=x_crop, params=params, method='leastsq', nan_policy='propagate')
    # print(fit.fit_report(min_correl=0.25))

    fit_stat = _save_fit_params(fit, fit_stat)

    df = _save_fit_in_df(df=df, fit=fit, column_to_fit=column_to_fit, column_extension=column_extension)
    return df, fit_stat


def _get_init_params(fct, init_params, model, x_crop, y_crop, i, number_unique_lorentzians=None):
    params = model.make_params()

    if init_params is None:
        # guess initial params
        if fct == 'poly_gaussian' or fct == 'poly_lorentzian' or fct == 'double_gaussian':
            raise UserWarning('please provide initial params; for poly_gaussian params guess is not yet implemented')

        if fct == 'lorentzian':
            init_params = {}
            init_params['amp'] = np.max(y_crop)
            init_params['cen'] = x_crop[np.argmax(y_crop)]
            init_params['off'] = np.max(x_crop)
            init_params['gamma'] = abs(init_params['cen']
                                       - x_crop[get_nearest_index_in_array(y_crop, (init_params['amp']
                                                                                    - init_params['off']) / 2)])
        if fct == 'gaussian':
            init_params = {}
            init_params['amp'] = np.max(y_crop)
            init_params['cen'] = x_crop[np.argmax(y_crop)]
            init_params['off'] = np.max(x_crop)
            init_params['sig'] = abs(init_params['cen']
                                     - x_crop[get_nearest_index_in_array(y_crop, (init_params['amp']
                                                                                  - init_params['off']) / 2)])
        if fct == 'poly_lorentzian':
            if number_unique_lorentzians is None:
                raise UserWarning('number of unique lorentzains need to be provided for the fit')

    if params.keys() == init_params.keys():
        for param in params.keys():
            params[param].set(init_params[param])
    else:
        raise UserWarning('provided and expected parameters do not match')

    return params


def _make_model(fct, number_unique_lorentzians=None):
    if not fct == poly_gaussian and not fct == double_gaussian and not fct == poly_lorentzian:
        return Model(fct, independent_vars=['x'])
    elif not fct == poly_lorentzian:
        return fct()
    else:
        return poly_lorentzian(number_unique_lorentzians=number_unique_lorentzians)


def gaussian(x, amp, cen, sig, off):
    return amp / (np.sqrt(2 * np.pi) * sig) * np.exp(-((x - cen) / sig) ** 2 / 2) + off


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


def lorentzian(x, amp, cen, gamma):
    return amp / (np.pi * gamma) * 1 / (1 + (x - cen) ** 2 / gamma ** 2)


def poly_lorentzian(number_unique_lorentzians=None):
    model = Model(lorentzian, independent_vars=['x'], prefix='lorentzian1_')
    model += LinearModel(independent_vars=['x'], prefix='linear1_')
    if number_unique_lorentzians is None:
        raise UserWarning('number_unique_lorentzians need to be given')
    for i in range(2, number_unique_lorentzians + 1):
        model += Model(lorentzian, independent_vars=['x'], prefix='lorentzian{}_'.format(i))
    return model


def _save_fit_in_df(df, fit, column_to_fit, column_extension=''):
    x_init = df.index.values
    df['Best fit - ' + column_to_fit + column_extension] = fit.eval(x=x_init)
    return df


def _save_fit_params(fit, fit_data):
    fit_params = fit.params
    fit_data['redchi'].append(fit.redchi)
    for key in fit_data.keys():
        if not key[-4:] == '_err' and not key == 'redchi' and not key == 'file':
            if key not in fit_params.keys():
                fit_data[key].append(0.0)
                fit_data[key + '_err'].append(0.0)
            else:
                fit_data[key].append(fit_params[key].value)
                fit_data[key + '_err'].append(fit_params[key].stderr)

    return fit_data


def create_fit_data_from_params(dfs, column_to_fit, fit_data, fct='gaussian', column_extension=''):
    new_dfs = []
    for df in dfs:
        df, file_name = df

        if fct == 'gaussian':
            amp = fit_data.loc[file_name[:4], 'amp']
            cen = fit_data.loc[file_name[:4], 'cen']
            sig = fit_data.loc[file_name[:4], 'sig']
            off = fit_data.loc[file_name[:4], 'off']

            x = df.index.values

            df['Best fit - ' + column_to_fit + column_extension] = pd.Series(data=gaussian(x, amp, cen, sig, off),
                                                                             index=df.index)
            new_dfs.append((df, file_name))
        else:
            raise UserWarning('create fit from params not yet implemented for the chosen function')
    return new_dfs
