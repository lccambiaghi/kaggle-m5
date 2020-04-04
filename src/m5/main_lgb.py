import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import plot_importance
import matplotlib.pyplot as plt
import neptune
from neptunecontrib.monitoring.lightgbm import neptune_monitor

from m5.global_model import get_X_and_y
from m5.hierarchy import compute_series_weights


data = Path('data/')
sales = pd.read_csv(data / 'sales_train_validation.csv')
prices = pd.read_csv(data / 'sell_prices.csv')
calendar = pd.read_csv(data / 'calendar.csv', parse_dates=['date'])

# cast to categories with same cat codes
sales['item_id'] = pd.Categorical(sales['item_id'], ordered=True)
sales['store_id'] = pd.Categorical(sales['store_id'], ordered=True)
prices['item_id'] = pd.Categorical(prices['item_id'], categories=sales['item_id'].cat.categories, ordered=True)
prices['store_id'] = pd.Categorical(prices['store_id'], categories=sales['store_id'].cat.categories, ordered=True)

cat_features = ['item_id', 'store_id',
            'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI',
            'is_year_end', 'is_year_start', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_weekend']

forecast_horizon = 28
(X_train, y_train), (X_val, y_val), X_test = get_X_and_y(sales, prices, calendar, cat_features, val_days = 90)

train_set = lgb.Dataset(X_train, y_train, categorical_feature = cat_features, free_raw_data=False)
val_set = lgb.Dataset(X_val, y_val, categorical_feature = cat_features, reference=train_set, free_raw_data=False)

NEPTUNE=True

PARAMS = {
    "num_leaves": 80,
    "seed": 23,
    "objective": "regression",
    "boosting_type": "gbdt",
    "min_data_in_leaf": 200,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "metric": "l2",
    "num_boost_round": 5000,
}

callbacks=[]
if NEPTUNE:
    neptune.set_project("lccambiaghi/kaggle-m5")
    neptune.create_experiment(
        name=f"lightgbm",
        send_hardware_metrics=False,
        run_monitoring_thread=False,
        params=PARAMS,
    )
    callbacks.append(neptune_monitor(prefix=f'h{forecast_horizon}_'))


model = lgb.train(PARAMS, train_set, early_stopping_rounds=125, verbose_eval=100,
                  valid_sets = [train_set, val_set], callbacks=callbacks)

model_filename = f'data/models/h{forecast_horizon}_lgbm.txt'
model.save_model(model_filename)
# log feature importance
fig, ax = plt.subplots()
plot_importance(model, max_num_features=20, figsize=(20,10), ax=ax)
importance_filename = f'plots/h{forecast_horizon}_lgbm_importance'
# save val predictions
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
y_pred_val = pd.DataFrame(y_pred_val, index=X_val.index, columns=['prediction'])
y_pred_val_filename = f'data/preds/h{forecast_horizon}_y_pred_val.parquet'
y_pred_val.to_parquet(output_filename)
# save test predictions
y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_test = pd.DataFrame(y_pred_test, index=X_test.index, columns=['prediction'])
y_pred_test_filename = f'data/preds/h{forecast_horizon}_y_pred_test.parquet'
y_pred_test.to_parquet(output_filename)

if NEPTUNE:
    neptune.log_metric(f"h{forecast_horizon}_val_rmse", val_rmse)
    neptune.log_artifact(model_filename)
    neptune.log_image(importance_filename, fig)
    neptune.log_artifact(y_pred_val_filename)
    neptune.log_artifact(y_pred_test_filename)
    neptune.stop()

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
