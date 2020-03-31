import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb

# import sys
# sys.path.append('src')
from m5.global_model import get_X_and_y
from m5.hierarchy import compute_series_weights


data = Path('data/')
sales = pd.read_csv(data / 'sales_train_validation.csv', dtype={
    'item_id':'category', 'dept_id':'category', 'cat_id':'category', 'store_id':'category', 'state_id':'category'
})
prices = pd.read_csv(data / 'sell_prices.csv', dtype={
    'store_id': 'category', 'item_id': 'category'
})
calendar = pd.read_csv(data / 'calendar.csv', parse_dates=['date'], dtype={
    'weekday':'category', 'd':'category', 'event_name':'category', 'event_type_1':'category', 'event_type_2':'category',
    'snap_CA':'category', 'snap_TX':'category', 'snap_WI':'category'
})

cat_features = ['item_id', 'store_id',
            'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI',
            'is_year_end', 'is_year_start', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_weekend']

X_train, y_train, X_val, y_val = get_X_and_y(sales.iloc[:100, :], prices, calendar, cat_features)

train_set = lgb.Dataset(X_train, y_train, categorical_feature = cat_features, free_raw_data=False)
val_set = lgb.Dataset(X_val, y_val, categorical_feature = cat_features, reference=train_set, free_raw_data=False)

params = {'metric': 'l2', 'objective': 'regression','seed': 23}

model = lgb.train(params, train_set, num_boost_round = 5000, early_stopping_rounds=100,
                  valid_sets = [train_set, val_set], verbose_eval = 100)
model.save_model('model.txt')

def get_y_weights(y: pd.Series, normalize=False):
    """
    For each series, compute the denominator in the MSSE loss function, i.e. the
    day-to-day variations squared, averaged by number of training observations.
    The weights can be normalized so that they add up to 1.
    This is provided to the lgb.Dataset for computing loss function and evaluation metric
    """
    scales = (y.unstack(level='date').diff(axis=1) ** 2).mean(axis=1)
    scales = scales.replace(0, pd.NA)
    weights = 1 / scales
    if normalize:
        weights = weights.divide(weights.sum())
    weights = y.merge(weights.to_frame('weight'), left_index=True, right_index=True)['weight']
    return weights

train_weights = get_y_weights(y_train)
val_weights = get_y_weights(y_val)

train_set = lgb.Dataset(X_train, y_train, weight=train_weights, categorical_feature = cat_features, free_raw_data=False)
val_set = lgb.Dataset(X_val, y_val, weight=val_weights, categorical_feature = cat_features, reference=train_set, free_raw_data=False)

def sse(preds, train_data):
    true = train_data.get_label()
    weights = train_data.get_weight() # weights is 1 / scale, normalized
    # loss = weights * (preds - true) ** 2
    gradient = weights * 2 * (preds - true)
    hessian = weights * 2
    return gradient, hessian

# model = lgb.train(params, train_set, num_boost_round = 5000, early_stopping_rounds=100,
#                   valid_sets = [train_set, val_set], verbose_eval = 100, fobj=sse)

weights = compute_series_weights(sales, prices, calendar, groups={})

def get_wrmsse_lgb(preds, data):
    if preds.shape[0] == y_val.shape[0]:
        y = y_val.copy()
    else:
        y = y_train.copy()

    y['prediction'] = preds
    y['weight'] = data.get_weight() # weights is 1 / scale, normalized
    y['sse'] = ((y['sales'] - y['prediction']) ** 2) * y['weight']

    rmsse = y[['sse']].unstack(level='date').mean(axis=1).pow(1/2).to_frame(name='rmsse')
    rmsse =  rmsse.merge(weights['series_weight'], left_index=True, right_index=True)
    wrmsse = rmsse['rmsse'] * rmsse['series_weight']

    score = wrmsse.sum()
    return 'wrmsse', score, False

model = lgb.train(params, train_set, num_boost_round = 5000, early_stopping_rounds=100,
                  valid_sets = [train_set, val_set], verbose_eval = 100, fobj=sse, feval=get_wrmsse_lgb)
model.save_model('model_custom_loss.txt')
