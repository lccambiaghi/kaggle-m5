import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_sales_and_prices_wide(sales, prices, calendar):
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
    # remove last 28 dates as we have no actuals for them
    prices_wide = prices_wide.iloc[:, :-28]
    # only consider prices for items in sales_wide
    prices_wide = prices_wide.loc[prices_wide.index.intersection(sales_wide.index), :]
    # set demand to NA in dates where a product was not available
    # sales_wide = sales_wide.where(prices_wide.notnull())
    # log.info(f'Replaced {np.count_nonzero(sales_wide.isna())} zero sales with NA')

    return sales_wide, prices_wide

def compute_sales_features(sales_wide):
    sales_features_wide = {}
    for lag in tqdm([0, 7, 28], "autoregressive features"):
        autoregressive = sales_wide.copy()
        autoregressive.columns = autoregressive.columns.shift(lag, freq='D')
        sales_features_wide[f"ar_{lag}"] = autoregressive

    for window in tqdm([7, 30, 60, 90, 180], "rolling features"):
        mean = sales_wide.rolling(window, axis=1).mean()
        sales_features_wide[f"mean_{window}"] = mean

        std = sales_wide.rolling(window, axis=1).std()
        sales_features_wide[f"std_{window}"] = std

        ewma = sales_wide.ewm(span=window).mean()
        sales_features_wide[f"ewma_{window}"] = ewma

    for window in tqdm([30], "skew, kurt features"):
        skew = sales_wide.rolling(window, axis=1).skew()
        sales_features_wide[f"skew_{window}"] = skew

        kurt = sales_wide.rolling(window, axis=1).kurt()
        sales_features_wide[f"kurt_{window}"] = kurt

    return sales_features_wide

def compute_prices_features(prices_wide):
    prices_features_wide = {}
    for window in tqdm([7, 30], "price std features"):
        price_std = prices_wide.rolling(window, axis=1).std()
        prices_features_wide[f"price_std_{window}"] = price_std

    for window in tqdm([365], "price change features"):
        price_max_year = prices_wide.rolling(window, axis=1).max()
        price_change_year = (prices_wide - price_max_year) / price_max_year
        price_change_week = (prices_wide - prices_wide.shift(1, axis=1)) / prices_wide.shift(1, axis=1)

        prices_features_wide[f"price_max_365"] = price_max_year
        prices_features_wide[f"price_change_365"] = price_change_year
        prices_features_wide[f"price_change_7"] = price_change_week

    return prices_features_wide

def stack_wide_features(features_wide, dates):
    features_long = []
    for _, f in tqdm(features_wide.items(), "stacking features"):
        # only select needed dates + casting before stacking is efficient
        f = f.loc[:, dates].astype(np.float32).stack(dropna=False).sort_index() # sort index so we are sure all features are aligned
        features_long.append(f.values[:, np.newaxis]) # convert 1D array to column vector
    features_long = pd.DataFrame(data=np.hstack(features_long), index=f.index, columns=[*features_wide.keys()])
    features_long.index = features_long.index.rename('date', level=-1)

    return features_long

def merge_fast(df_left, df_right):
    """
    This method merges two dataframes ignoring the index and it is much faster than pd.concat()
    If all the columns are of the same dtype, they are preserved.
    For this reason, float32 is good common_type.
    """
    return pd.DataFrame(data=np.c_[df_left.values, df_right.values], index = df_left.index, columns = df_left.columns.tolist() + df_right.columns.tolist())

def compute_datetime_features(dates, calendar, cat_features):
    datetime_features = calendar[['date', 'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI']].set_index('date').merge(pd.DataFrame(index=dates),
                                                                                                                                    left_index=True, right_index=True)
    datetime_attrs = [
        "quarter",
        "month",
        "week",
        "day",
        "year",
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
    datetime_features.loc[:, "is_weekend"] = datetime_features.index.dayofweek.isin([5, 6])

    for c in [c for c in datetime_features if c in cat_features]:
        datetime_features[c] = datetime_features[c].astype('category').cat.codes

    # cast them to float32 for merge_fast to work
    datetime_features = datetime_features.astype('float32')

    return datetime_features

def cast_features(X, cat_features):
    int_8_cols = [c for c in ["quarter", "month", "week","day","dayofweek"] if c not in cat_features]
    X[int_8_cols] = X[int_8_cols].astype('int8')

    X['year'] = X['year'].astype('int32')
    X['date'] = X['date'].astype('int64')

    # cast back from codes to categories
    X[cat_features] = X[cat_features].astype('category')

    return X

def build_design_matrix(rolling_features, prices_features, calendar, cat_features, train_days, val_dates, test_dates, forecast_horizon):
    for name, f in rolling_features.items():
        f.columns = f.columns.shift(forecast_horizon, freq='D')

    train_val_test_dates = list(prices_features.values())[0].columns
    # only consider dates for which all features have at least one observation
    for name, f in rolling_features.items():
        train_val_test_dates = train_val_test_dates.intersection(f.dropna(axis=1, how='all').columns)
        assert all(val_dates.union(test_dates).isin(train_val_test_dates)), f'{name} is always NA for val or test dates'

    train_dates = train_val_test_dates.difference(test_dates).difference(val_dates)[-train_days:]
    log.info(f'Using {len(train_dates)} train days, {len(val_dates)} val days, {len(test_dates)} test days')

    X_dict = {}
    for name, dates in {'train': train_dates, 'val': val_dates, 'test': test_dates}.items():
        X = stack_wide_features(prices_features, dates)
        rolling_features_long = stack_wide_features(rolling_features, dates)
        X = merge_fast(X, rolling_features_long)

        X_dates = X.index.get_level_values('date')
        datetime_features = compute_datetime_features(X_dates, calendar, cat_features)
        X = merge_fast(X, datetime_features)

        # set index as columns to add item_id, store_id and date
        X = X.assign(**X.index.to_frame())
        X = cast_features(X, cat_features)

        X_dict[name]=X

    return (*X_dict.values()), train_dates


def get_X_and_y(sales: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame, cat_features: List[str], forecast_horizon=28, train_days=1000, val_days=180):
    sales_wide, prices_wide = get_sales_and_prices_wide(sales, prices, calendar)

    rolling_features = compute_sales_features(sales_wide)
    prices_features = compute_prices_features(prices_wide)

    test_dates, val_dates = prices_wide.columns[-28:][:forecast_horizon], prices_wide.columns[-val_days-28:-28]

    # shift rolling features according to forecast horizon
    X_train, X_val, X_test, train_dates = build_design_matrix(rolling_features, prices_features, calendar, cat_features, train_days, val_dates, test_dates, forecast_horizon)

    y_train, y_val = sales_wide.loc[:, train_dates].stack(dropna=False).sort_index(), sales_wide.loc[:, val_dates].stack(dropna=False).sort_index()
    y_train.index, y_val.index = y_train.index.rename('date', level=-1), y_val.index.rename('date', level=-1)

    # only keep observations where all numerical features are not null
    non_null = np.where(X_train.select_dtypes('number').notna().all(1))[0]
    log.info(f'Keeping {100*len(non_null)/len(X_train)}% train observations')
    X_train, y_train = X_train.iloc[non_null], y_train.iloc[non_null]

    non_null = np.where(X_val.select_dtypes('number').notna().all(1))[0]
    log.info(f'Keeping {100*len(non_null)/len(X_val)}% val observations')
    X_val, y_val = X_val.iloc[non_null], y_val.iloc[non_null]

    non_null = np.where(X_test.select_dtypes('number').notna().all(1))[0]
    log.info(f'Keeping {100*len(non_null)/len(X_test)}% test observations')
    X_test = X_test.iloc[non_null]

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert X_test.notnull().all().all()

    return (X_train, y_train), (X_val, y_val), X_test
