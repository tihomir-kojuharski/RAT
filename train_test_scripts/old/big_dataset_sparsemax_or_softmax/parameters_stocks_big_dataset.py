import os

from utils.train_test_utils import Parameters, DatasetParameters

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

random_assets = len(assets)
params_stocks_big_dataset = Parameters(
    dataset=DatasetParameters(
        dataset=os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../../data/stocks_daily/stocks_big_dataset_2022-0425.nc')),
        x_window_size=30,
        batch_size=128,
        dataset_date_range=slice('2006-06-01', None),
        # dataset_features=None,
        # dataset_features=['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'signal_line'],
        # dataset_features=['close', 'high', 'low', 'open'],
        dataset_assets=assets,
        train_range=('2010-05-01', '2020-05-01'),
        validation_range=('2020-05-01', '2021-05-01'),
        # test_range=('2007-10-01', '2008-12-01'),
        test_range=('2021-05-01', None),
        assets_per_batch=random_assets
        # test_range=('2021-05-01', None)
    ),

    model_dim=len(assets),
    output_dir=f'../../../output_old/stocks_big_dataset',
    total_step=35000,
    output_step=50,
    multihead_num=12,
    local_context_length=7,
    encoder_decoder_layers=6,
    activation="softmax",

    # dataset_features=['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'signal_line'],
    test_dataset_assets=assets,
    random_train_assets=random_assets,

    trading_consumption=0.0025,
    variance_penalty=0.0,
    cost_penalty=0.0,
    learning_rate=1e-4,
    weight_decay=1e-7,
    daily_interest_rate=0.001)
