from utils.train_test_utils import Parameters

params_snp500 = Parameters(
    dataset='../data/snp500/all_stocks_5yr.nc',
    output_dir = f'../output/snp',
    total_step=20000,
    x_window_size=7,
    batch_size=16,
    feature_number=4,
    output_step=50,
    multihead_num=1,
    local_context_length=3,
    encoder_decoder_layers=1,

    test_portion=0.08,
    validation_portion=0.08,
    trading_consumption=0.0025,
    variance_penalty=0.0,
    cost_penalty=0.0,
    learning_rate=0.0001,
    weight_decay=1e-7,
    daily_interest_rate=0.001)
