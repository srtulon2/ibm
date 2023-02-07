from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score

from smogn import phi,phi_ctrl_pts

import pandas as pd

import numpy as np

def calculate_phi(y, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):
    # relevance method, method ("auto" or "manual")
    # distribution focus, xtrm_type ("high", "low", "both")
    # coefficient for box plot, coef
    # input for "manual" rel method, ctrl_pts
    phi_params = phi_ctrl_pts(y = y,method = method, xtrm_type = xtrm_type, coef = coef, ctrl_pts = ctrl_pts)
    y_phi = phi(y = y,ctrl_pts = phi_params)

    return y_phi

def root_mse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def root_mae(y, y_pred):
    return np.sqrt(mean_absolute_error(y, y_pred))

def phi_weighted_r2(y, y_pred):
    y_phi=calculate_phi(y)
    return r2_score(y, y_pred, sample_weight=y_phi)

def phi_weighted_mse(y, y_pred):
    y_phi=calculate_phi(y)
    return mean_squared_error(y, y_pred, sample_weight=y_phi)

def phi_weighted_mae(y, y_pred):
    y_phi=calculate_phi(y)
    return mean_absolute_error(y, y_pred, sample_weight=y_phi)

def phi_weighted_root_mse(y, y_pred):
    y_phi=calculate_phi(y)
    return np.sqrt(mean_squared_error(y, y_pred, sample_weight=y_phi))

def phi_weighted_root_mae(y, y_pred):
    y_phi=calculate_phi(y)
    return np.sqrt(mean_absolute_error(y, y_pred, sample_weight=y_phi))

def sera (trues, preds, step = 0.001,return_err = False) :
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)
    
    phi_trues= calculate_phi(trues)

    trues = trues.values
    tbl = pd.DataFrame(
        {'trues': trues,
         'phi_trues': phi_trues,
         })
    tbl = pd.concat([tbl, preds], axis=1)
    ms = list(tbl.columns[2:])
    th = np.arange(0, 1 + step, step)
    errors = []
    for ind in th:
        errors.append(
            [sum(tbl.apply(lambda x: ((x['trues'] - x[y]) ** 2) if x['phi_trues'] >= ind else 0, axis=1)) for y in ms])

    areas = []
    for x in range(1, len(th)):
        areas.append([step *(errors[x - 1][y] + errors[x][y]) / 2 for y in range(len(ms))])
    areas = pd.DataFrame(data=areas, columns=ms)
    res = areas.apply(lambda x: sum(x))
    if return_err :
       return {"sera":res, "errors":errors, "thrs" :th}
    else:
       return res.item()