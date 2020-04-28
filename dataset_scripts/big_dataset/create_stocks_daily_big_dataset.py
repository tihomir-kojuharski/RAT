import os

import numpy as np
import pandas as pd
import xarray as xr


datadir = "../../data/stocks_daily/stocks_big_dataset_2022-0425"
output_file = f"{datadir}.nc"

# IMPORTANT: The order of the features is important - the model looks for the first feature to be the close price!
features = ['close', 'high', 'low', 'open', 'volume', 'unadjusted_close', 'pe_ratio_ttm', 'eps_surprise_percentage']
symbols = []
times = set()

frequency_minutes = 30

symbol_dfs = {}

for symbol in os.listdir(datadir):
    symbols.append(symbol)

    earnings_df = pd.read_csv(f"{datadir}/{symbol}/earnings.csv", parse_dates=['reportedDate', 'fiscalDateEnding'])
    earnings_df_orig = earnings_df.copy()

    earnings_df.sort_values(['reportedDate', 'fiscalDateEnding'], inplace=True)
    earnings_df.set_index('reportedDate', inplace=True)

    earnings_df = earnings_df.groupby(earnings_df.index).agg('last')

    if len(earnings_df):
        # days_diff = (np.roll(earnings_df.index, -1) - earnings_df.index).total_seconds() / (
        #         3600 * 24)
        # assert len(days_diff[days_diff > 140]) == 0
        earnings_df.loc[earnings_df['reportedEPS'] == 'None', 'reportedEPS'] = np.nan
        earnings_df.loc[earnings_df['surprisePercentage'] == 'None', 'surprisePercentage'] = np.nan
        earnings_df['surprisePercentage'] = earnings_df['surprisePercentage'].astype(float)

        earnings_df['eps_ttm'] = earnings_df.rolling(4)['reportedEPS'].sum()

        earnings_df = earnings_df.resample('1D') \
            .ffill() \
            .rename(
            columns={'reportedEPS': 'eps', 'estimatedEPS': 'estimated_eps',
                     'surprisePercentage': 'eps_surprise_percentage'}) \
            [['eps_ttm', 'eps_surprise_percentage']]
        earnings_df.index.name = 'time'
    else:
        earnings_df = pd.DataFrame(columns=['eps_ttm', 'eps_surprise_percentage'])

    prices_df = pd.read_csv(f"{datadir}/{symbol}/adjusted_prices.csv", parse_dates=['time'],
                            index_col='time')
    prices_df.sort_values('time', inplace=True)

    symbol_df = prices_df.join(earnings_df, how='left')
    adjustment_coeff = symbol_df['adjusted_close'] / symbol_df['close']

    symbol_df.rename(columns={'close': 'unadjusted_close'}, inplace=True)
    symbol_df.rename(columns={'adjusted_close': 'close'}, inplace=True)

    for feature in ['open', 'high', 'low']:
        symbol_df[feature] = symbol_df[feature] * adjustment_coeff

    symbol_df['eps_ttm'] = symbol_df['eps_ttm'].ffill()
    symbol_df['eps_surprise_percentage'] = symbol_df['eps_surprise_percentage'].ffill()
    symbol_df['pe_ratio_ttm'] = symbol_df['close'] / symbol_df['eps_ttm']

    symbol_dfs[symbol] = symbol_df

    times.update(symbol_df.index.tolist())

times = sorted(map(lambda t: t.to_pydatetime(), times))

da = xr.DataArray(coords=[features, symbols, times], dims=['feature', 'asset', 'time'])

for symbol in symbols:
    symbol_df = symbol_dfs[symbol]

    for feature in features:
        da.loc[feature, symbol, symbol_df.index.tolist()] = symbol_df[feature]

da.to_netcdf(output_file)
