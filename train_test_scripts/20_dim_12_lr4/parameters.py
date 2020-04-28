import os.path

from utils.train_test_utils import Parameters, DatasetParameters

assets = ['XOM', 'GE', 'MSFT', 'C', 'T', 'BAC', 'PG', 'WMT', 'PFE', 'MO', 'AIG', 'JNJ']

# random_assets = len(assets)
random_assets = None

params = Parameters(
    dataset=DatasetParameters(
        dataset=os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../data/stocks_daily/stocks_big_dataset_2022-0425.nc')),
        x_window_size=30,
        batch_size=128,
        dataset_date_range=slice('2006-06-01', None),
        # dataset_features=None,
        # dataset_features=['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'signal_line'],
        dataset_features=['close', 'high', 'low', 'open'],
        dataset_assets=assets,
        train_range=('2010-05-01', '2020-05-01'),
        validation_range=('2020-05-01', '2021-05-01'),
        test_range=('2007-10-01', '2008-12-01'),
        assets_per_batch=random_assets
        # test_range=('2021-05-01', None)
    ),

    output_dir=f'../../output/',
    total_step=100000000,

    output_step=20,
    multihead_num=12,
    local_context_length=5,
    encoder_decoder_layers=1,
    model_dim=4,
    d_encoder_decoder_embedding=24,
    activation="softmax",

    trading_consumption=0.0025,
    variance_penalty=0.0,
    cost_penalty=0.0,
    learning_rate=1e-4,
    weight_decay=1e-7,
    daily_interest_rate=0.001,

    save_last_model=True,

    random_train_assets=random_assets,
    test_dataset_assets=assets,
)
