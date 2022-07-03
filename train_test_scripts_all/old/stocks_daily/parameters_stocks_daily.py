from utils.train_test_utils import Parameters

params_stocks_daily = Parameters(
    dataset='../data/stocks_daily/stocks_big_dataset_2022-0425.nc',
    output_dir=f'../output/stocks_daily',
    total_step=20000,
    x_window_size=30,
    batch_size=128,
    output_step=20,
    multihead_num=12,
    local_context_length=7,
    encoder_decoder_layers=6,

    dataset_date_range=slice('2010-06-29', None),
    dataset_features=['close', 'open', 'high', 'low'],
    dataset_assets=[
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
    ],

    validation_date='2020-05-01',
    test_date='2021-05-01',
    trading_consumption=0.0025,
    variance_penalty=0.0,
    cost_penalty=0.0,
    learning_rate=1e-4,
    weight_decay=1e-7,
    daily_interest_rate=0.001)
