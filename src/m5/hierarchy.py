from typing import Dict, List
import numpy as np
import pandas as pd


all_names = ['level', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']
all_groups = {'total': ['d'],
        'state': ['d', 'state_id'], 'cat': ['d', 'cat_id'], 'item': ['d', 'item_id'],
        'store': ['d', 'state_id', 'store_id'], 'state/cat':['d', 'state_id', 'cat_id'], 'dept': ['d', 'cat_id', 'dept_id'], 'item/state': ['d', 'item_id', 'state_id'],
        'store/cat': ['d', 'state_id', 'store_id', 'cat_id'], 'dept/state': ['d', 'cat_id', 'dept_id', 'state_id'],
        'store/dept': ['d', 'state_id', 'store_id', 'cat_id', 'dept_id']}


def get_sales_long(sales: pd.DataFrame, groups: Dict[str, List[str]] = all_groups, include_bottom_level=True, last_28_days=False):
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    if last_28_days:
        d_cols = d_cols[-28:]

    if groups:
        sales_melted = sales.melt(id_vars=all_names[1:], value_vars=d_cols, var_name='d', value_name='sales')

    all_sales_grouped = []
    for level, group in groups.items():
        sales_grouped = sales_melted.groupby(group, as_index=False).sum()

        missing = [c for c in all_names if c not in group]
        for m in missing:
            sales_grouped.loc[:, m] = 'all'
        sales_grouped.loc[:, 'level'] = level

        sales_grouped = sales_grouped.set_index(['d'] + all_names).unstack(all_names)
        sales_grouped.columns = sales_grouped.columns.droplevel(0)
        all_sales_grouped.append(sales_grouped)

    if include_bottom_level:
        sales_grouped = sales[all_names[1:] + d_cols]
        sales_grouped.loc[:, 'level'] = 'item/store'
        sales_grouped = sales_grouped.set_index(all_names).T
        all_sales_grouped.insert(0, sales_grouped)

    all_sales_grouped = pd.concat(all_sales_grouped, axis=1)

    return all_sales_grouped

def compute_summing_matrix(sales: pd.DataFrame, groups=all_groups.copy()):
    sales['series_id'] = sales.reset_index().index

    # bottom submatrix
    sales['level'] = 'item/store'
    sales_grouped = sales.set_index(all_names)
    bottom_level = sales_grouped.index
    bottom_submatrix = pd.DataFrame(data=np.identity(sales.shape[0]), index=sales_grouped.index, columns=bottom_level)

    # top row
    groups.pop('total', None)
    sales_grouped = sales.copy()
    for m in all_names:
        sales_grouped.loc[:, m] = 'all'
    group_index = sales_grouped.groupby(all_names)['series_id'].agg(list).index
    top_row = pd.DataFrame(data=np.ones((1,sales.shape[0])), index=group_index, columns=bottom_level)

    # central submatrices
    central_submatrices = []
    for level, level_names in groups.items():
        sales_grouped = sales.groupby(level_names[1:], as_index=False)[['series_id']].agg(list)

        level_rows = []
        for _, group in sales_grouped['series_id'].iteritems():
            row = np.zeros(sales.shape[0])
            row[group] = 1
            level_rows.append(row)

        missing = [c for c in all_names if c not in group]
        for m in missing:
            sales_grouped.loc[:, m] = 'all'
        sales_grouped.loc[:, 'level'] = level
        group_index = sales_grouped.set_index(all_names).index

        level_submatrix = pd.DataFrame(data=np.vstack(level_rows), index=group_index, columns=bottom_level)
        central_submatrices.append(level_submatrix)

    central_submatrices = pd.concat(central_submatrices)
    summing_matrix = pd.concat([top_row, central_submatrices, bottom_submatrix])

    return summing_matrix

def get_prices_long(prices: pd.DataFrame, calendar: pd.DataFrame, sales: pd.DataFrame, groups: Dict[str, List[str]] = all_groups, include_bottom_level=True, last_28_days=True):
    prices_d = prices.merge(calendar[['wm_yr_wk', 'd']]).drop(columns=['wm_yr_wk'])
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    if last_28_days:
        prices_d = prices_d[prices_d['d'].isin(d_cols[-28:])]
    prices_d = sales[all_names[1:]].merge(prices_d)
    prices_d.loc[:, 'level'] = 'item/store'

    all_prices_h = []
    for level, group in groups.items():
        prices_h = prices_d.groupby(group, as_index=False).sum()

        missing = [c for c in all_names if c not in group]
        for m in missing:
            prices_h.loc[:, m] = 'all'
        prices_h.loc[:, 'level'] = level

        prices_h = prices_h.set_index(['d'] + all_names).unstack(all_names)
        prices_h.columns = prices_h.columns.droplevel(0)
        all_prices_h.append(prices_h)

    if include_bottom_level:
        prices_h = prices_d.set_index(['d'] + all_names).unstack(all_names)
        prices_h.columns = prices_h.columns.droplevel(0)
        all_prices_h.insert(0, prices_h)

    all_prices_h = pd.concat(all_prices_h, axis=1)
    return all_prices_h

def compute_series_weights(sales: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame, groups: Dict[str, List[str]] = all_groups, include_bottom_level=True):
    prices_long = get_prices_long(prices, calendar, sales, groups=groups, include_bottom_level=include_bottom_level, last_28_days=True)
    sales_long = get_sales_long(sales, groups=groups, include_bottom_level=include_bottom_level, last_28_days=True)

    assert all(sales_long.columns == prices_long.columns)
    assert all(sales_long.index == prices_long.index)

    dollar_value_by_series = (prices_long * sales_long).sum(axis=0).to_frame(name='series_dollar_value')
    dollar_value_by_level = (prices_long * sales_long).sum(axis=0).sum(level='level').to_frame(name='level_dollar_value')
    dollar_value = dollar_value_by_series.merge(dollar_value_by_level, left_index=True, right_index=True)
    dollar_value.loc[:, 'series_dollar_value_scaled'] = dollar_value['series_dollar_value'] / dollar_value['level_dollar_value']
    assert dollar_value['series_dollar_value_scaled'].sum(level='level').all() == 1

    dollar_value.loc[:, 'series_weight'] = dollar_value['series_dollar_value_scaled'] / (1 + len(groups))
    return dollar_value

def get_sales_long_with_nan(sales, prices, calendar, groups={}):
    prices_long = get_prices_long(prices, calendar, sales, groups=groups, last_28_days=False)
    sales_long = get_sales_long(sales, groups=groups)
    assert all(sales_long.columns == prices_long.columns)

    prices_long = prices_long.loc[sales_long.index]
    assert all(sales_long.index == prices_long.index)

    sales_long[prices_long.isna()] = pd.NA
    return sales_long
