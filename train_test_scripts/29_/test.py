import copy

import parameters
from utils.train_test_utils import run_test

test_params = copy.copy(parameters.params)


# test_params.test_dataset_assets =  ['C', 'AIG', 'GE', 'XOM', 'MSFT', 'T', 'BAC', 'PG', 'WMT', 'PFE', 'MO', 'JNJ']
# test_params.dataset.test_range = ('2007-10-01', '2008-12-01')
test_params.dataset.test_range = ('2021-05-01', None)
# test_params.dataset.test_range = ('2020-05-01', '2021-04-30')
# test_params.dataset.test_range = ('2010-05-03', '2020-04-30')


if __name__ == "__main__":
    run_test(run_dir="../../output/20220507_223531", params=test_params, test_type='test', test_model='last')

