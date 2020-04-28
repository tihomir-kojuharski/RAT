import logging
import os

import pandas as pd
import time
from datetime import datetime

from alpha_vantage.timeseries import TimeSeries

from utils.alpha_vantage_addons import FundamentalDataWithEarnings

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

API_KEY = ""

ts = TimeSeries(key=API_KEY, output_format='pandas')
fd = FundamentalDataWithEarnings(key=API_KEY, output_format='json')

snp500_df = pd.read_csv("../data/snp500/all_stocks_5yr.csv")

symbols = sorted(snp500_df['Name'].unique().tolist())
for symbol in ['ADS', 'AGN', 'APC', 'BBT', "BF.B"]:
    symbols.remove(symbol)

snp500_stocks = ['A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK',
                 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT',
                 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'APD',
                 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX',
                 'BBWI', 'BBY', 'BDX', 'BEN', 'BF.B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL', 'BMY', 'BR',
                 'BRK.B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI',
                 'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CEG', 'CERN', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF',
                 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST',
                 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CTXS',
                 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISH', 'DLR',
                 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY',
                 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'ES', 'ESS',
                 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FB', 'FBHS',
                 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRC', 'FRT',
                 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN',
                 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON',
                 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN',
                 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J',
                 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI',
                 'KMX', 'KO', 'KR', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW',
                 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK',
                 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH',
                 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
                 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NLOK', 'NLSN', 'NOC', 'NOW', 'NRG', 'NSC',
                 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC',
                 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PEAK', 'PEG', 'PENN', 'PEP', 'PFE',
                 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL',
                 'PRU', 'PSA', 'PSX', 'PTC', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN',
                 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBNY', 'SBUX',
                 'SCHW', 'SEDG', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE',
                 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL',
                 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN',
                 'TT', 'TTWO', 'TWTR', 'TXN', 'TXT', 'TYL', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP',
                 'UPS', 'URI', 'USB', 'V', 'VFC', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ',
                 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK',
                 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS']

symbols = snp500_stocks + [
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

for symbol in ["BF.B"]:
    symbols.remove(symbol)

requests_per_minute = 75

current_time_str = datetime.now().strftime("%Y-%m%-d")

datadir = f"../data/stocks_daily/stocks_big_dataset_{current_time_str}"

os.makedirs(datadir, exist_ok=True)

for symbol in symbols:
    requests_per_symbol = 0

    symbol_dir = f"{datadir}/{symbol}"
    os.makedirs(symbol_dir, exist_ok=True)

    logging.info(f"Dumping data for {symbol}")
    earnings_file = f"{symbol_dir}/earnings.csv"
    if not os.path.exists(earnings_file):
        requests_per_symbol += 1
        logging.info("earnings...")

        try:
            earnings, _ = fd.get_earnings(symbol)
            earnings_df = pd.DataFrame(earnings["quarterlyEarnings"])
        except ValueError as e:
            earnings_df = pd.DataFrame(columns=['fiscalDateEnding', 'reportedDate', 'reportedEPS', 'estimatedEPS',
                                                'surprise', 'surprisePercentage'])

        earnings_df.to_csv(earnings_file, index=False)

    adjusted_prices_file = f"{symbol_dir}/adjusted_prices.csv"
    if not os.path.exists(adjusted_prices_file):
        requests_per_symbol += 1

        logging.info("adjusted prices...")
        adjusted_prices_df, _ = ts.get_daily_adjusted(symbol, outputsize='full')
        adjusted_prices_df.index.name = 'time'
        adjusted_prices_df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                                           '5. adjusted close': 'adjusted_close', '6. volume': 'volume',
                                           '7. dividend amount': 'dividend_amount',
                                           '8. split coefficient': 'split_coefficient'},
                                  inplace=True)
        adjusted_prices_df.to_csv(adjusted_prices_file)

    if requests_per_symbol > 0:
        time.sleep(((60.0 / requests_per_minute) * requests_per_symbol) + 1)
