import os.path

from utils.train_test_utils import Parameters

assets = [
    # 'SPY',  # S&P 500
    'GE',
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

params = Parameters(
    dataset=os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../data/stocks_daily/stocks_big_dataset_2022-0425.nc')),
    output_dir=f'../../output/stocks_big_dataset',
    total_step=100000,
    x_window_size=30,
    batch_size=128,
    output_step=50,
    multihead_num=12,
    local_context_length=7,
    encoder_decoder_layers=6,
    activation="softmax",

    dataset_date_range=slice('2010-06-29', None),
    # dataset_features=['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'signal_line'],
    test_dataset_assets=assets,
    dataset_assets=assets,
    # random_train_assets=12,

    validation_date='2020-05-01',
    test_date='2021-05-01',
    trading_consumption=0.0025,
    variance_penalty=0.0,
    cost_penalty=0.0,
    learning_rate=1e-5,
    weight_decay=1e-7,
    daily_interest_rate=0.001)
