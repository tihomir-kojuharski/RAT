from parameters_snp500 import params_snp500
from utils.train_test_utils import run_test

if __name__ == "__main__":
    run_test(run_dir="../output/snp/20220421_142735", params=params_snp500, test_type='validation')
