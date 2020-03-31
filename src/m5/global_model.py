import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List


def get_sales_and_price_wide(sales, prices, calendar):
    # map days to dates
    day_to_date = calendar.set_index("d")["date"].to_dict()
    sales_wide = sales.rename(columns=day_to_date)
    # select dates
    all_dates = sales_wide.select_dtypes('int').columns
    sales_wide = sales_wide.set_index(['item_id', 'store_id'])[all_dates]
    sales_wide.columns = pd.to_datetime(sales_wide.columns).rename('date')
    # add date to prices
    prices_wide = prices.merge(calendar[['date', 'wm_yr_wk']]).drop(columns=['wm_yr_wk']).set_index(['date', 'item_id', 'store_id']).unstack(level='date').sort_index()
    prices_wide.columns = prices_wide.columns.droplevel(0)
    prices_wide.columns = pd.to_datetime(prices_wide.columns).rename('date')
    # make sure prices_wide and sales_wide have same index
    # TODO what about prices for future dates?
    prices_wide = prices_wide.loc[prices_wide.index.intersection(sales_wide.index), sales_wide.columns.intersection(sales_wide.columns)]
    # set demand to NA in dates where a product was not available
    sales_wide[prices_wide.isna()] = pd.NA

    return prices_wide, sales_wide

def compute_float_features(sales_wide, prices_wide, forecast_horizon = 28):
    float_features_wide = {}
    for lag in tqdm([0, 7, 28], "autoregressive features"):
        autoregressive = sales_wide.shift(lag, axis=1)
        float_features_wide[f"ar_{lag}"] = autoregressive

    for window in tqdm([7, 28, 56, 91], "rolling features"):
        mean = sales_wide.rolling(window, axis=1).mean()
        float_features_wide[f"mean_{window}_h{forecast_horizon}"] = mean

        std = sales_wide.rolling(window, axis=1).std()
        float_features_wide[f"std_{window}_h{forecast_horizon}"] = std

        ewma = sales_wide.ewm(span=window, min_periods=window).mean()
        float_features_wide[f"ewma_{window}_h{forecast_horizon}"] = ewma

    for window in tqdm([28], "skew, kurt features"):
        skew = sales_wide.rolling(window, axis=1).skew()
        float_features_wide[f"skew_{window}_h{forecast_horizon}"] = skew

        kurt = sales_wide.rolling(window, axis=1).kurt()
        float_features_wide[f"kurt_{window}_h{forecast_horizon}"] = kurt

    for window in tqdm([7, 28], "price std features"):
        price_std = prices_wide.rolling(window, axis=1).std()
        float_features_wide[f"price_std_{window}"] = price_std

    for window in tqdm([365], "price change features"):
        price_max_year = prices_wide.rolling(window, axis=1).max()
        price_change_year = (prices_wide - price_max_year) / price_max_year
        price_change_week = (prices_wide - prices_wide.shift(1, axis=1)) / prices_wide.shift(1, axis=1)

        float_features_wide[f"price_max_365"] = price_max_year
        float_features_wide[f"price_change_365"] = price_change_year
        float_features_wide[f"price_change_7"] = price_change_week

    return float_features_wide

def get_train_val_dates(float_features_wide, val_days, only_val):
    train_val_dates = list(float_features_wide.values())[0].columns
    # only include dates for which all features have at least one observation
    for _, f in tqdm(float_features_wide.items(), "computing train and validation index"):
        train_val_dates = train_val_dates.intersection(f.dropna(axis=1, how='all').columns)

    train_dates = train_val_dates[:-val_days] if not only_val else pd.DatetimeIndex([])
    val_dates = train_val_dates[-val_days:]

    return train_dates, val_dates

def stack_float_features(float_features_wide, train_dates, val_dates):
    train_val_dates = train_dates.union(val_dates)

    float_features_names = [*float_features_wide.keys()]
    all_features_long = []
    for name in tqdm(float_features_names, "building design matrix"):
        f = float_features_wide.pop(name)  # casting before stacking is efficient
        f = f.loc[:, train_val_dates].astype(np.float32).stack(dropna=False).sort_index() # sort index so we are sure all features are aligned
        all_features_long.append(f.values[:, np.newaxis]) # convert 1D array to column vector
    all_features_long = pd.DataFrame(data=np.hstack(all_features_long), index=f.index, columns=float_features_names)
    all_features_long.index = all_features_long.index.rename('date', level=-1)

    return all_features_long

def add_datetime_features(train_dates, val_dates, calendar):
    train_val_dates = train_dates.union(val_dates)

    datetime_features = calendar[['date', 'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI']].set_index('date').loc[train_val_dates]

    datetime_attrs = [
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]
    for attr in tqdm(datetime_attrs, "datetime features"):
        datetime_features.loc[:, attr] = getattr(datetime_features.index, attr)
        # cast int features to int8, the others are categorical
        if not attr.startswith('is'):
            datetime_features[attr] = datetime_features[attr].astype('int8')
    datetime_features.loc[:, "is_weekend"] = datetime_features.index.dayofweek.isin([5, 6])
    datetime_features.loc[:, "year"] = datetime_features.index.year

    return datetime_features

def get_X_and_y(sales: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame, cat_features: List[str], val_days=28, only_val=False):
    sales_wide, prices_wide = get_sales_and_price_wide(sales, prices, calendar)

    float_features_wide = compute_float_features(sales_wide, prices_wide)

    train_dates, val_dates = get_train_val_dates(float_features_wide, val_days, only_val)
    # test_dates = prices_wide.iloc[:, -28:]

    all_features_long = stack_float_features(float_features_wide, train_dates, val_dates)

    # set index as columns to add item_id, store_id and date
    all_features_long = all_features_long.assign(**all_features_long.index.to_frame())

    datetime_features = add_datetime_features(train_dates, val_dates, calendar)
    all_features_long = all_features_long.merge(datetime_features, left_index=True, right_index=True)
    all_features_long["date"] = all_features_long["date"].astype('int')

    # cast categories
    all_features_long[cat_features] = all_features_long[cat_features].astype('category')

    # TODO add function that computes X_val and X_test by shifting with the forecast horizon
    # TODO dropna on num features
    sales_long = sales_wide.loc[:, train_dates.union(val_dates)].stack(dropna=False).sort_index()
    sales_long.index = sales_long.index.rename('date', level=-1)

    y_train, X_train = sales_long.loc(axis=0)[:, :, train_dates], all_features_long.loc(axis=0)[:, :, train_dates]
    y_val, X_val = sales_long.loc(axis=0)[:, :, val_dates], all_features_long.loc(axis=0)[:, :, val_dates]
    # X_test = all_features_long.loc(axis=0)[:, :, test_dates]

    return (X_train, y_train), (X_val, y_val)#, X_test
