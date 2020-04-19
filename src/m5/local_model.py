import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from joblib import Parallel, delayed
from collections import defaultdict
import argh

from m5.hierarchy import get_sales_long_with_nan, get_sales_long
from m5.features import add_special_event_feature, add_snap_feature
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_exogenous_features(calendar):
    calendar = add_snap_feature(calendar)
    calendar = add_special_event_feature(calendar)

    features = ['special_event', 'snap_CA', 'snap_TX', 'snap_WI', 'snap_all']
    X = calendar[['date'] + features].set_index('date').iloc[:-28]  # remove last 28 days until June

    # remove chistmas weeks as they pollute the models
    christmas_weeks = calendar[calendar.event_name_1 == 'Christmas']['wm_yr_wk']
    chistmas_weeks_dates = calendar[calendar['wm_yr_wk'].isin(christmas_weeks)]['date']
    X = X[~X.index.isin(chistmas_weeks_dates)]

    return X

def get_X_and_y(series: pd.Series, X: pd.DataFrame):
    train_dates, test_dates = X.index[:-28], X.index[-28:]
    y_train = series.loc[train_dates].dropna()

    state = y_train.name[1]
    features = ['special_event', f'snap_{state}']
    X_train, X_test = X[features].loc[y_train.index], X[features].loc[test_dates]

    return y_train, X_train, X_test

def get_cv_score(model, y, X):
    n_initial = len(y.loc[:'3-Jan-16'])
    cv_score = model.cv_score(y, X, max_horizon=28, n_initial=n_initial, step=7)
    # return average RMSE across all 28 horizons
    return cv_score.values.mean()

def forecast_best_model(model, y_train, X_train, X_test):
    model.fit(y_train, X_train)
    y_pred = model.predict(X_test.shape[0], X_test)
    model.model = None

    y_pred = pd.Series(y_pred['prediction'].values, index=X_test.index, name=y_train.name)
    return y_pred


def cv_and_forecast_intermittent_series(y, X):
    from rpy2_models.ts_model_cv import ESXCV
    model = ESXCV()

    y_train, X_train, X_test = get_X_and_y(y, X)
    X_train = pd.concat([X_train, X_test])

    try:
        score = get_cv_score(model, y_train, X_train)
        y_pred = forecast_best_model(model, y_train, X_train, X_test)
    except Exception as e:
        print(f"'{e}' when calculating cv_score for {y_train.name}")
        return y_train.name, {}

    return y_train.name, {'y_pred': y_pred, 'best_model': model}

def forecast_all_bottom_series(sales: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame):
    sales_long = get_sales_long_with_nan(sales, prices, calendar)
    day_to_date = calendar.set_index("d")["date"].to_dict()
    sales_long = sales_long.rename(index=day_to_date)

    X = get_exogenous_features(calendar)

    all_forecasts = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
        delayed(cv_and_forecast_intermittent_series)(y, X) for _, y in sales_long.iteritems()
    )

    joblib.dump(dict(all_forecasts), 'data/preds/y_pred_intermittent.joblib')

def cv_and_forecast_grouped_series(y, X):
    from rpy2_models.ts_model_cv import ExponentialSmoothingCV, ESXCV, ARIMAXCV
    y_train, X_train, X_test = get_X_and_y(y, X)

    results = {}
    try:
        ets_model = ExponentialSmoothingCV()
        ets_score = get_cv_score(ets_model, y_train, X_train)
        results['ets_score'] = ets_score.astype('float32')
        results['ets_errors'] = ets_model.cv_errors.astype('float32')
        y_pred = forecast_best_model(ets_model, y_train, X_train, X_test)
        results['ets_forecast'] = y_pred

        arimax_model = ARIMAXCV(seasonal=True)
        arimax_score = get_cv_score(arimax_model, y_train, X_train)
        results['arimax_score'] = arimax_score.astype('float32')
        results['arimax_errors'] = arimax_model.cv_errors.astype('float32')
        y_pred = forecast_best_model(arimax_model, y_train, X_train, X_test)
        results['arimax_forecast'] = y_pred
    except Exception as e:
        print(f"'{e}' when calculating cv_score for {y_train.name}")
        return y_train.name, results

    return y_train.name, results

def forecast_all_grouped_series(sales: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame):
    sales_long = get_sales_long(sales, include_bottom_level=False)
    day_to_date = calendar.set_index("d")["date"].to_dict()
    sales_long = sales_long.rename(index=day_to_date)

    X = get_exogenous_features(calendar)

    all_forecasts = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
        delayed(cv_and_forecast_grouped_series)(y, X) for _, y in sales_long.iteritems()
    )

    joblib.dump(dict(all_forecasts), 'data/preds/y_pred_grouped.joblib')

@argh.arg('forecast_level', choices=['intermittent', 'grouped'])
def choose_forecast_level(forecast_level):
    data = Path('data/')
    sales = pd.read_csv(data / 'sales_train_validation.csv')
    prices = pd.read_csv(data / 'sell_prices.csv')
    calendar = pd.read_csv(data / 'calendar.csv', parse_dates=['date'])

    if forecast_level == 'intermittent':
        forecast_all_bottom_series(sales, prices, calendar)
    if forecast_level == 'grouped':
        forecast_all_grouped_series(sales, prices, calendar)

if __name__ == '__main__':
    argh.dispatch_command(choose_forecast_level)
