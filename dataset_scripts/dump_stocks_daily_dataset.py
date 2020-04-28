import logging
import os

import pandas as pd
import time
from datetime import datetime

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

API_KEY = ""
BASE_URL = 'https://www.alphavantage.co/query'
FUNCTION_NAME = 'TIME_SERIES_DAILY'

symbols = [
    'SPY',  # S&P 500
    'DAL',  # Delta Airlines
    'AAPL',  # Apple
    'AMD',  # AMD
    'DVN',  # Devon Energy Corp
    'INTC',  # Intel Corp
    'JPM',  # JP Morgan
    'IBM',  # Pinterest
    'RIO',  # Rio Tinto
    'TEVA',  # Teva
    'PCG',  # PG&E Corporation
    'TSLA',  # Tesla
]

requests_per_minute = 5

current_time_str = datetime.now().strftime("%Y-%m%-d")

datadir = f"../data/stocks/12_stocks_daily_{current_time_str}"

os.makedirs(datadir, exist_ok=True)

for symbol in symbols:
    csv_file = f"{datadir}/{symbol}.csv"
    if os.path.exists(csv_file):
        logging.info(f"Data for stock {symbol} has already been dumped, so will skip it in this run")
        continue

    logging.info(f"Dumping data for {symbol}")

    symbol_df = pd.read_csv(
        f'{BASE_URL}?function={FUNCTION_NAME}&symbol={symbol}&outputsize=full&datatype=csv&apikey={API_KEY}')

    symbol_df.rename({'timestamp': 'time'}, axis=1, inplace=True)
    symbol_df.sort_values('time', inplace=True)
    symbol_df.to_csv(csv_file, index=False)

    time.sleep((60.0 / requests_per_minute) + 1)

