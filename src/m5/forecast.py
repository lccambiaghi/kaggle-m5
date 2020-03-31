import pandas as pd
import numpy as np

from m5.features import add_demand_type_feature
from m5.hierarchy import get_sales_long_with_nan, all_names
from rpy2_models.ts_model import SExpS, Croston
from rpy2_models import r_models_functions


def get_y_train_val(series: pd.Series, val_days=28):
    y_train = series.dropna()
    y_val = series.iloc[-val_days:]
    return y_train, y_val


def forecast_bottom_level(sales: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame):
    sales_with_type = add_demand_type_feature(sales.copy())
    sales_with_type = sales_with_type.set_index(all_names[1:])

    sales_long = get_sales_long_with_nan(sales, prices, calendar)

    y_pred_long = []
    for _, series in sales_long.iteritems():
        series_name = series.name[1:]
        series_type = sales_with_type.loc[series_name]['type']
        if series_type in ['lumpy', 'intermittent']:
            model = Croston(type='sba')
        elif series_type == 'erratic':
            model = Croston(type='optimized')
        else:
            model = SExpS()

        y_train, y_val = get_y_train_val(series)
        y_pred = model.fit_predict(y_train.values, y_val, None, None)
        y_pred_long.append(y_pred)

    y_pred_long = pd.concat(y_pred_long, axis=1)
    return y_pred_long

def get_rmsse(y_hat: pd.DataFrame, sales_long: pd.DataFrame):
    y_true = sales_long.loc[y_hat.index, y_hat.columns]
    y_train = sales_long.loc[sales_long.index.difference(y_hat.index), y_hat.columns]

    mse = ((y_hat - y_true) ** 2).mean(axis=0)
    scale = ((y_train.diff(axis=0)) ** 2).mean(axis=0)
    rmsse = np.sqrt(mse / scale)
    return rmsse

def get_wrmsse(y_hat: pd.DataFrame, sales_long: pd.DataFrame, weights: pd.DataFrame):
    rmsse = get_rmsse(y_hat, sales_long)
    wrmsse = (weights.loc[rmsse.index]['series_weight'] * rmsse).sum()
    return wrmsse
