import os

from pgportfolio.tools.indicator import max_drawdown
from utils.train_test_utils import load_dataset, DatasetParameters

assets = ['XOM', 'GE', 'MSFT'] #, 'C', 'T', 'BAC', 'PG', 'WMT', 'PFE', 'MO', 'AIG', 'JNJ']

dataset_params = DatasetParameters(
    dataset=os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../data/stocks_daily/stocks_big_dataset_2022-0425.nc')),
    x_window_size=30,
    batch_size=128,
    dataset_date_range=slice('2006-06-01', None),
    dataset_features=['close', 'high', 'low', 'open'],
    dataset_assets=assets,
    train_range=('2010-06-29', '2020-05-01'),
    validation_range=('2020-05-01', '2021-05-01'),
    # test_range=('2007-10-01', '2008-12-01')
    test_range=('2021-05-03', '2022-04-23')
    # test_range=('2010-05-03', '2020-04-30')
    # test_range=('2020-05-01', '2021-05-01'),
)

dm = load_dataset(dataset_params)

test_indices = dm._test_ind


dataset = dm.get_test_set_online(assets, test_indices[0], test_indices[-1]+1, 30, get_last_w_omega=False)
gains = dataset['y'][:, 0, :]

apv = gains.prod(axis=2).mean()
gain = apv - 1

tst_pc_array = gains.mean(axis=1).reshape(-1)
sr_rewards = tst_pc_array - 1
SR = sr_rewards.mean() / sr_rewards.std()
SN = apv
MDD = max_drawdown(tst_pc_array)
CR = SN / MDD

print(f"Gain: {gain*100:.2f}%")
print(f"APV: {apv}")
print(f"Sharpe Ratio: {SR}")
print(f"Calmar Ratio: {CR}")

