from train_test_scripts.parameters_stocks_daily import params_stocks_daily
from utils.train_test_utils import run_training, setup_logging, setup_output_dir

if __name__ == "__main__":
    run_training(params_stocks_daily)
