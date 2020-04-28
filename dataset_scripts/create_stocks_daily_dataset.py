import os
from datetime import datetime, timedelta, date
import pandas as pd
import xarray as xr

datadir = "../data/stocks/12_stocks_daily_2022-0421"
output_file = f"{datadir}.nc"

# IMPORTANT: The order of the features is important - the model looks for the first feature to be the close price!
features = ['close', 'high', 'low', 'open',  'volume']
symbols = []
times = set()

frequency_minutes = 30

for filename in os.listdir(datadir):
    symbol = filename[:filename.find(".")]
    symbols.append(symbol)
    symbol_df = pd.read_csv(f"{datadir}/{filename}")

    symbol_times = symbol_df['time'].tolist()

    if len(times) == 0:
        times = set(symbol_times)
    else:
        times = times.intersection(symbol_times)

min_time = min(times)

times = set()
for filename in os.listdir(datadir):
    symbol = filename[:filename.find(".")]

    symbol_times = symbol_df[symbol_df['time'] >= min_time]['time'].tolist()
    times.update(symbol_times)

times = sorted(map(lambda t: datetime.fromisoformat(t), times))

da = xr.DataArray(coords=[features, symbols, times], dims=['feature', 'asset', 'time'])

for filename in os.listdir(datadir):
    symbol = filename[:filename.find(".")]
    symbol_df = pd.read_csv(f"{datadir}/{filename}", parse_dates=['time'])
    valid_times = symbol_df['time'].isin(times)

    for feature in features:
        da.loc[feature, symbol, symbol_df[valid_times]['time'].tolist()] = symbol_df[valid_times][feature]

da.to_netcdf(output_file)
