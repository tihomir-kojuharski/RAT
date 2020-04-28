from parameters_stocks_daily import params_stocks_daily
from utils.train_test_utils_before_refactoring import run_test

if __name__ == "__main__":
    run_test(run_dir="../output/stocks_daily/20220427_081325", params=params_stocks_daily, test_type='validation')
