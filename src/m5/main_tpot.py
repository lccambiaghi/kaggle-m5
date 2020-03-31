import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tpot import TPOTRegressor
from joblib import dump

import sys
sys.path.append('src')
from m5.global_model import compute_features, get_X_y

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

sales_long = compute_features(sales, prices, calendar)

cat_features = ['item_id', 'store_id',
            'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI',
            'is_year_end', 'is_year_start', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_weekend']

X_train, y_train, X_val, y_val = get_X_y(sales_long, cat_features)

model = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42, config_dict='TPOT light')

model.fit(X_train, y_train)

dump(model, 'tpot.joblib')
print(model.export())
