import copy

import parameters
from utils.train_test_utils import run_test

test_params = copy.copy(parameters.params)

if __name__ == "__main__":
    run_test(run_dir="../../output/20220503_212928", params=test_params, test_type='validation')
