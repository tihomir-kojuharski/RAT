import copy

import parameters
from utils.train_test_utils import run_test

test_params = copy.copy(parameters.params)


# test_params.test_dataset_assets =  ['C', 'AIG', 'GE', 'XOM', 'MSFT', 'T', 'BAC', 'PG', 'WMT', 'PFE', 'MO', 'JNJ']
test_params.dataset.test_range = ('2007-10-01', '2008-12-01')
# test_params.dataset.test_range = ('2021-05-01', '2022-04-23')
# test_params.dataset.test_range = ('2020-05-01', '2021-05-01')
# test_params.dataset.test_range = ('2010-05-03', '2020-05-30')


if __name__ == "__main__":
    run_test(run_dir="../../output/40_20220509_124142", params=test_params, test_type='test', test_model='step_850000')
    # run_test(run_dir="../../output/40_20220509_124142", params=test_params, test_type='test', test_model='step_1400000')

