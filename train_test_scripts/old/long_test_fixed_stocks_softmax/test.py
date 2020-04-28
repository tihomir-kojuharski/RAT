from train_test_scripts.long_test_fixed_stocks_softmax.parameters import params
from utils.train_test_utils import run_test

if __name__ == "__main__":
    run_test(run_dir="../../output/stocks_big_dataset/20220502_193728", params=params, test_type='validation')
