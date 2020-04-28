from train_test_scripts.stocks_daily.parameters_stocks_daily import params_stocks_daily
from utils.train_test_utils import load_dataset

dm = load_dataset(params_stocks_daily)

dataset = dm.get_test_set(normalized=True)

result = dataset['y'][:, 0, :].prod(axis=0).mean()
gain = result - 1

print(f"Gain: {gain:.2f}%")

pass
