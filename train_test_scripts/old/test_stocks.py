from parameters_stocks import params_stocks
from utils.train_test_utils import run_test

if __name__ == "__main__":
    run_test(run_dir="../output/snp/20220419_233953", params=params_stocks, test_type='validation')
