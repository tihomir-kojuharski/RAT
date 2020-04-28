from dataclasses import dataclass
from datetime import datetime

import torch

from loss import SimpleLossCompute, Batch_Loss, SimpleLossCompute_tst, Test_Loss
from pgportfolio.marketdata.datamatricesnew import DataMatricesNew
from utils.train_test_utils import Parameters

params_stocks = Parameters(
    dataset='../data/stocks/12_stocks_30min_2022-0419.nc',
    output_dir=f'../output/stocks',
    total_step=20000,
    x_window_size=50,
    batch_size=128,
    output_step=50,
    multihead_num=2,
    local_context_length=5,
    encoder_decoder_layers=1,

    test_portion=0.08,
    validation_portion=0.08,
    trading_consumption=0.0025,
    variance_penalty=0.0,
    cost_penalty=0.0,
    learning_rate=0.0001,
    weight_decay=1e-7,
    daily_interest_rate=0.001)
