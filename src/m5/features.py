import pandas as pd
import numpy as np


def add_snap_feature(calendar: pd.DataFrame):
    snap_cols = [col for col in calendar.columns if col.startswith('snap_')]
    calendar['snap'] = calendar[snap_cols].sum(axis=1)
    return calendar

def add_special_event_feature(calendar: pd.DataFrame):
    calendar['special_event'] = 0
    calendar.loc[calendar['event_type_1'].notna(), 'special_event'] = 1
    return calendar

def add_cv2_feature(sales: pd.DataFrame):
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    _sales = sales.replace({0: pd.NA})
    sales['cv2'] = (_sales[d_cols].std(axis=1) / _sales[d_cols].mean(axis=1)) ** 2
    return sales

def add_adi_feature(sales: pd.DataFrame):
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    sales['adi'] = sales.shape[1] / np.count_nonzero(sales[d_cols], axis=1)
    return sales

def add_demand_type_feature(sales: pd.DataFrame):
    sales = add_cv2_feature(sales)
    sales = add_adi_feature(sales)

    sales.loc[(sales['adi']>4/3) & (sales['cv2']>0.5), 'type'] = 'lumpy'
    sales.loc[(sales['adi']>4/3) & (sales['cv2']<=0.5), 'type'] = 'intermittent'
    sales.loc[(sales['adi']<=4/3) & (sales['cv2']>0.5), 'type'] = 'erratic'
    sales.loc[(sales['adi']<=4/3) & (sales['cv2']<=0.5), 'type'] = 'smooth'
    return sales
