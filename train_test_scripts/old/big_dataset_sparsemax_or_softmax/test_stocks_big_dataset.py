from train_test_scripts.old.big_dataset_sparsemax_or_softmax.parameters_stocks_big_dataset import \
    params_stocks_big_dataset
from utils.train_test_utils import run_test

if __name__ == "__main__":
    run_test(run_dir="../../../output_old/stocks_big_dataset/20220429_161139", params=params_stocks_big_dataset, test_type='test')
