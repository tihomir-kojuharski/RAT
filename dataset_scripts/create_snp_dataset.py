import os
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr

input_file = "../data/snp500/all_stocks_5yr.csv"
output_file = "../data/snp500/all_stocks_5yr.nc"

# IMPORTANT: The order of the features is important - the model looks for the first feature to be the close price!
features = ['close', 'high', 'low', 'open',  'volume']
times = set()

df = pd.read_csv(input_file, parse_dates=['date'])
# df['date'] = df['date'].dt.date

symbols = sorted(df['Name'].unique().tolist())
dates = pd.Series(df['date'].unique()).sort_values().tolist()

da = xr.DataArray(coords=[features, symbols, dates], dims=['feature', 'asset', 'time'])

for symbol in symbols:
    symbol_df = df[df['Name'] == symbol]

    for feature in features:
        da.loc[feature, symbol, symbol_df['date'].tolist()] = symbol_df[feature]

da.to_netcdf(output_file)
