import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple

import torch
import time
import numpy as np
import pandas as pd

from loss import SimpleLossCompute, Batch_Loss, Test_Loss, SimpleLossCompute_tst
from pgportfolio.marketdata.datamatricesnew import DataMatricesNew
from rat.rat import make_model


def train_one_step(DM, x_window_size, model, loss_compute, local_context_length, device, random_test_assets,
                   custom_commission_loss):
    batch = DM.next_batch(custom_commission_loss=custom_commission_loss,
                          random_assets=random_test_assets)
    batch_input = batch["X"]  # (128, 4, 11, 31)
    batch_y = batch["y"]  # (128, 4, 11)
    batch_last_w = batch["last_w"]  # (128, 11)
    batch_w = batch["setw"]
    #############################################################################
    previous_w = torch.tensor(batch_last_w, dtype=torch.float)
    previous_w = previous_w.to(device)
    previous_w = torch.unsqueeze(previous_w, 1)  # [128, 11] -> [128,1,11]
    batch_input = batch_input.transpose((1, 0, 2, 3))
    batch_input = batch_input.transpose((0, 1, 3, 2))
    src = torch.tensor(batch_input, dtype=torch.float)
    src = src.to(device)
    price_series_mask = (torch.ones(src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]
    currt_price = src.permute((3, 1, 2, 0))  # [4,128,31,11]->[11,128,31,4]
    if (local_context_length > 1):
        padding_price = currt_price[:, :, -(local_context_length) * 2 + 1:-1, :]
    else:
        padding_price = None
    currt_price = currt_price[:, :, -1:, :]  # [11,128,31,4]->[11,128,1,4]
    trg_mask = make_std_mask(currt_price, src.size()[1])
    batch_y = batch_y.transpose((0, 2, 1))  # [128, 4, 11] ->#[128,11,4]
    trg_y = torch.tensor(batch_y, dtype=torch.float)
    trg_y = trg_y.to(device)

    out = model.forward(src, currt_price, previous_w, price_series_mask, trg_mask, padding_price)

    new_w = out[:, :, 1:]  # 去掉cash
    new_w = new_w[:, 0, :]  # #[109,1,11]->#[109,11]
    new_w = new_w.detach().cpu().numpy()
    batch_w(new_w)

    loss, portfolio_value = loss_compute(out, trg_y, previous_w)
    if torch.isnan(loss).sum() > 0:
        logging.info("Found nan loss in the following batch:")
        logging.info(batch)
        exit(1)
    return loss, portfolio_value


def test_online(DM, test_indices, x_window_size, model, evaluate_loss_compute, local_context_length, device,
                test_dataset_assets, custom_commission_loss):
    tst_batch = DM.get_test_set_online(test_dataset_assets, test_indices[0], test_indices[-1] + 1, x_window_size,
                                       get_last_w_omega=custom_commission_loss)
    tst_batch_input = tst_batch["X"]
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = np.zeros_like(tst_batch["last_w"])
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float)
    tst_previous_w = tst_previous_w.to(device)
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)

    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))

    long_term_tst_src = torch.tensor(tst_batch_input, dtype=torch.float)
    long_term_tst_src = long_term_tst_src.to(device)
    #########################################################################################
    tst_src_mask = (torch.ones(long_term_tst_src.size()[1], 1, x_window_size) == 1)

    long_term_tst_currt_price = long_term_tst_src.permute((3, 1, 2, 0))
    long_term_tst_currt_price = long_term_tst_currt_price[:, :, x_window_size - 1:, :]
    ###############################################################################################
    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:, :, 0:1, :], long_term_tst_src.size()[1])

    tst_batch_y = tst_batch_y.transpose((0, 3, 2, 1))
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float)
    tst_trg_y = tst_trg_y.to(device)
    tst_long_term_w = []
    tst_y_window_size = len(test_indices) - x_window_size - 1
    prev_w = []

    for j in range(tst_y_window_size + 1):  # 0-9
        tst_src = long_term_tst_src[:, :, j:j + x_window_size, :]
        tst_currt_price = long_term_tst_currt_price[:, :, j:j + 1, :]
        if (local_context_length > 1):
            padding_price = long_term_tst_src[:, :,
                            j + x_window_size - 1 - local_context_length * 2 + 2:j + x_window_size - 1, :]
            padding_price = padding_price.permute((3, 1, 2, 0))  # [4, 1, 2, 11] ->[11,1,2,4]
        else:
            padding_price = None

        prev_w.append(tst_previous_w)

        out = model.forward(tst_src, tst_currt_price, tst_previous_w,
                            # [109,1,11]   [109, 11, 31, 3]) torch.Size([109, 11, 3]
                            tst_src_mask, tst_trg_mask, padding_price)
        if (j == 0):
            tst_long_term_w = out.unsqueeze(0)  # [1,109,1,12]
        else:
            tst_long_term_w = torch.cat([tst_long_term_w, out.unsqueeze(0)], 0)
        out = out[:, :, 1:]  # 去掉cash #[109,1,11]

        tst_previous_w = out

        # y = tst_trg_y[:, j, :, 0]
        # y_with_cash = torch.concat([torch.ones((y.shape[0], 1)), y], -1)  # [128,13]
        # last_w_with_cash = torch.concat([(1 - tst_previous_w.sum(axis=-1, keepdims=True)), tst_previous_w],
        #                                   -1)  # [128, 13]
        # tst_previous_w = tst_previous_w * y / (last_w_with_cash * y_with_cash).sum(-1).squeeze(-1)



    tst_long_term_w = tst_long_term_w.permute(1, 0, 2, 3)  ##[10,128,1,12]->#[128,10,1,12]
    tst_loss, portfolio_value_history, rewards, SR, CR, tst_pc_array, TO = evaluate_loss_compute(tst_long_term_w,
                                                                                                 tst_trg_y,
                                                                                                 torch.concat(prev_w, axis=0))
    return tst_loss, portfolio_value_history, rewards, SR, CR, tst_pc_array, TO, tst_long_term_w, tst_trg_y


def test_net(DM, test_indices, x_window_size, local_context_length, model, device,
             trading_consumption, interest_rate, variance_penalty, cost_penalty, test_dataset_assets,
             custom_commission_loss):
    "Standard Training and Logging Function"
    start = time.time()
    ####每个epoch开始时previous_w=0

    test_loss_compute = SimpleLossCompute_tst(
        Test_Loss(trading_consumption, interest_rate, device, custom_commission_loss, variance_penalty, cost_penalty))

    #########################################################tst########################################################
    with torch.no_grad():
        model.eval()
        tst_loss, portfolio_value_history, rewards, SR, CR, \
        tst_pc_array, TO, tst_long_term_w, tst_trg_y = test_online(
            DM, test_indices, x_window_size, model, test_loss_compute, local_context_length, device,
            test_dataset_assets, custom_commission_loss)
        elapsed = time.time() - start
        logging.info("Test Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
                     (
                         tst_loss.item(), portfolio_value_history[-1].item(), SR.item(), CR.item(), TO.item(),
                         1 / elapsed))
        start = time.time()
        #                portfolio_value_list.append(portfolio_value.item())

        log_SR = SR
        log_CR = CR
        log_tst_pc_array = tst_pc_array
    return portfolio_value_history, rewards, log_SR, log_CR, log_tst_pc_array, TO, tst_long_term_w, tst_trg_y


# def test_episode(DM, x_window_size, model, evaluate_loss_compute, local_context_length, device):
#     test_set = DM.get_validation_set()
#     test_set_input = test_set["X"]  # (TEST_SET_SIZE, 4, 11, 31)
#     test_set_y = test_set["y"]
#
#     test_previous_w = torch.zeros([1, 1, test_set_input.shape[2]])
#
#     losses = []
#     portfolio_values = []
#
#     for i in range(len(test_set_input)):
#         tst_batch_input = test_set_input[i:i + 1]  # (1, 4, 11, 31)
#
#         test_batch_y = test_set_y[i:i + 1]
#
#         test_previous_w = test_previous_w.to(device)
#         # test_previous_w = torch.unsqueeze(test_previous_w, 1)  # [2426, 1, 11]
#         tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
#         tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))
#         tst_src = torch.tensor(tst_batch_input, dtype=torch.float)
#         tst_src = tst_src.to(device)
#         tst_src_mask = (torch.ones(tst_src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]
#         tst_currt_price = tst_src.permute((3, 1, 2, 0))  # (4,128,31,11)->(11,128,31,3)
#         #############################################################################
#         if (local_context_length > 1):
#             padding_price = tst_currt_price[:, :, -(local_context_length) * 2 + 1:-1, :]  # (11,128,8,4)
#         else:
#             padding_price = None
#         #########################################################################
#
#         tst_currt_price = tst_currt_price[:, :, -1:, :]  # (11,128,31,4)->(11,128,1,4)
#         tst_trg_mask = make_std_mask(tst_currt_price, tst_src.size()[1])
#         test_batch_y = test_batch_y.transpose((0, 2, 1))  # (128, 4, 11) ->(128,11,4)
#         tst_trg_y = torch.tensor(test_batch_y, dtype=torch.float)
#         tst_trg_y = tst_trg_y.to(device)
#         ###########################################################################################################
#         tst_out = model.forward(tst_src, tst_currt_price, test_previous_w,  # [128,1,11]   [128, 11, 31, 4])
#                                 tst_src_mask, tst_trg_mask, padding_price)
#
#         tst_loss, tst_portfolio_value = evaluate_loss_compute(tst_out, tst_trg_y)
#
#         tst_loss = tst_loss.item()
#         losses.append(tst_loss)
#         portfolio_values.append(tst_portfolio_value)
#
#         # exclude the cash
#         test_previous_w = tst_out[:, :, 1:]
#
#     return np.mean(losses), np.prod(portfolio_values)


def test_batch(DM, x_window_size, model, evaluate_loss_compute, local_context_length, device, test_dataset_assets,
               custom_commission_loss):
    tst_batch = DM.get_validation_set(test_dataset_assets, normalized=True, get_last_w_omega=custom_commission_loss)
    tst_batch_input = tst_batch["X"]  # (128, 4, 11, 31)
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float)
    tst_previous_w = tst_previous_w.to(device)
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)  # [2426, 1, 11]

    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))

    tst_src = torch.tensor(tst_batch_input, dtype=torch.float)
    tst_src = tst_src.to(device)

    tst_src_mask = (torch.ones(tst_src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]

    tst_currt_price = tst_src.permute((3, 1, 2, 0))  # (4,128,31,11)->(11,128,31,3)
    #############################################################################
    if (local_context_length > 1):
        padding_price = tst_currt_price[:, :, -(local_context_length) * 2 + 1:-1, :]  # (11,128,8,4)
    else:
        padding_price = None
    #########################################################################

    tst_currt_price = tst_currt_price[:, :, -1:, :]  # (11,128,31,4)->(11,128,1,4)
    tst_trg_mask = make_std_mask(tst_currt_price, tst_src.size()[1])
    tst_batch_y = tst_batch_y.transpose((0, 2, 1))  # (128, 4, 11) ->(128,11,4)
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float)
    tst_trg_y = tst_trg_y.to(device)
    ###########################################################################################################
    tst_out = model.forward(tst_src, tst_currt_price, tst_previous_w,  # [128,1,11]   [128, 11, 31, 4])
                            tst_src_mask, tst_trg_mask, padding_price)

    tst_loss, tst_portfolio_value = evaluate_loss_compute(tst_out, tst_trg_y, tst_previous_w)
    return tst_loss, tst_portfolio_value


def train_net(DM, total_step, output_step, batch_size, x_window_size, local_context_length, model, output_dir, device,
              learning_rate, weight_decay, interest_rate, trading_consumption, variance_penalty, cost_penalty,
              random_test_assets, test_dataset_assets, save_last_model, continue_training_from, gradient_clipping_max,
              custom_commission_loss, validation_online, save_model_steps):
    #################set learning rate###################
    # lr_model_sz = 5120
    # warmup = 0  # 800
    #
    # model_opt = NoamOpt(lr_model_sz, learning_rate, warmup,
    #                     torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9,
    #                                      weight_decay=weight_decay))

    model_opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9,
                                  weight_decay=weight_decay)

    # model_opt = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    loss_compute = SimpleLossCompute(
        Batch_Loss(trading_consumption, interest_rate, device, custom_commission_loss, variance_penalty, cost_penalty,
                   True),
        model, gradient_clipping_max, model_opt)

    if validation_online:
        evaluate_loss_compute = SimpleLossCompute_tst(
            Test_Loss(trading_consumption, interest_rate, device, custom_commission_loss, variance_penalty,
                      cost_penalty))
    else:
        evaluate_loss_compute = SimpleLossCompute(
            Batch_Loss(trading_consumption, interest_rate, device, custom_commission_loss, variance_penalty, cost_penalty,
                       False), None)


    "Standard Training anlong_term_tst_currt_priced Logging Function"
    start = time.time()
    # total_loss = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value = 0

    if continue_training_from is not None:
        log = pd.read_csv(f"{output_dir}/train_log.csv").to_dict('records')
        starting_step = log[-1]['epoch']
    else:
        log = []
        starting_step = 0
    running_train_loss = 0
    running_train_apv = 0

    # validation_ds_size = len(DM.get_validation_set(test_dataset_assets, normalized=True)['y'])
    # train_loss_adjustment_coef = validation_ds_size / batch_size

    for i in range(starting_step, total_step):
        model.train()
        loss, portfolio_value = train_one_step(DM, x_window_size, model, loss_compute, local_context_length, device,
                                               random_test_assets, custom_commission_loss)
        running_train_loss += loss.item()
        running_train_apv += portfolio_value.item()
        # total_loss += loss.item()

        if ((i + 1) % output_step == 0):
            elapsed = time.time() - start
            train_loss = (running_train_loss / output_step)  # * train_loss_adjustment_coef
            train_apv = (running_train_apv / output_step)  # * train_loss_adjustment_coef

            logging.info("Epoch Step: %d| Train loss: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                         (i + 1, train_loss, train_apv, output_step / elapsed))
            start = time.time()
            #########################################################tst########################################################
            with torch.no_grad():
                model.eval()

                if validation_online:
                    tst_loss, portfolio_value_history, rewards, SR, CR, \
                    tst_pc_array, TO, tst_long_term_w, tst_trg_y = test_online(
                        DM, DM._validation_ind, x_window_size, model, evaluate_loss_compute, local_context_length, device,
                        test_dataset_assets, custom_commission_loss)
                    tst_portfolio_value = portfolio_value_history[-1]
                else:
                    tst_loss, tst_portfolio_value = test_batch(DM, x_window_size, model, evaluate_loss_compute,
                                                               local_context_length, device, test_dataset_assets,
                                                               custom_commission_loss)

                elapsed = time.time() - start
                logging.info("Test: %d Loss: %f| Portfolio_Value: %f | testset per Sec: %f \r\n" %
                             (i + 1, tst_loss.item(), tst_portfolio_value.item(), 1 / elapsed))
                start = time.time()

                log.append({
                    "time": datetime.now().isoformat(sep=' ', timespec='seconds'),
                    "epoch": i + 1,
                    "train_loss": train_loss,
                    "train_apv": train_apv,
                    "test_loss": tst_loss.item(),
                    "test_apv": tst_portfolio_value.item()
                })

                pd.DataFrame(log).to_csv(f"{output_dir}/train_log.csv", index=False)

                if save_last_model:
                    torch.save(model, f"{output_dir}/last_model.pkl")
                    logging.info("saved last model")

                if tst_portfolio_value > max_tst_portfolio_value:
                    max_tst_portfolio_value = tst_portfolio_value
                    torch.save(model, f"{output_dir}/best_model.pkl")
                    logging.info("saved best model!")

            running_train_loss = 0
            running_train_apv = 0

        if ((i + 1) % save_model_steps) == 0:
            torch.save(model, f"{output_dir}/step_{i + 1}.pkl")

    return tst_loss, tst_portfolio_value


def make_std_mask(local_price_context, batch_size):
    "Create a mask to hide padding and future words."
    local_price_mask = (torch.ones(batch_size, 1, 1) == 1)
    local_price_mask = local_price_mask & (subsequent_mask(local_price_context.size(-2)).type_as(local_price_mask.data))
    return local_price_mask


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class NoamOpt:
    "Optim wrapper that implements rate."

    # 512, 1, 400
    def __init__(self, model_size, learning_rate, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.learning_rate = learning_rate
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0:
            return self.learning_rate
        else:
            return self.learning_rate * \
                   (self.model_size ** (-0.5) *
                    min(step ** (-0.5), step * self.warmup ** (-1.5)))


@dataclass
class DatasetParameters:
    dataset: str
    x_window_size: int
    batch_size: int
    test_portion: Optional[float] = None
    validation_portion: Optional[float] = None
    train_range: Optional[Tuple[str, str]] = None
    validation_range: Optional[Tuple[str, str]] = None
    test_range: Optional[Tuple[str, str]] = None
    dataset_date_range: Optional[slice] = None
    dataset_assets: Optional[List[str]] = None
    dataset_features: Optional[List[str]] = None
    assets_per_batch: Optional[int] = None


@dataclass
class Parameters:
    dataset: DatasetParameters
    output_dir: str
    total_step: int
    output_step: int
    multihead_num: int
    local_context_length: int
    encoder_decoder_layers: int
    model_dim: int
    activation: str

    trading_consumption: float
    variance_penalty: float
    cost_penalty: float
    learning_rate: float
    weight_decay: float
    daily_interest_rate: float
    gradient_clipping_max: Optional[float] = None
    custom_commission_loss: Optional[bool] = False
    validation_online: Optional[bool] = False

    continue_training_from: Optional[str] = None
    random_train_assets: Optional[int] = None
    test_dataset_assets: Optional[List[str]] = None

    d_encoder_decoder_embedding: Optional[int] = None

    save_last_model: Optional[bool] = False
    save_model_steps: Optional[int] = 10000
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = "cpu"

    @property
    def interest_rate(self):
        return self.daily_interest_rate / 24 / 2


def load_dataset(params: DatasetParameters):
    return DataMatricesNew(dataset_file=params.dataset,
                           window_size=params.x_window_size,
                           is_permed=False,
                           buffer_bias_ratio=5e-5,
                           batch_size=params.batch_size,  # 128,
                           validation_portion=params.validation_portion,
                           test_portion=params.test_portion,
                           dataset_assets=params.dataset_assets,

                           dataset_date_range=params.dataset_date_range,
                           dataset_features=params.dataset_features,
                           train_range=params.train_range,
                           validation_range=params.validation_range,
                           test_range=params.test_range,

                           assets_per_batch=params.assets_per_batch,
                           portion_reversed=False)


def setup_logging(output_dir: Optional[str] = None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        log_file_handler = logging.FileHandler(f"{output_dir}/training.log")
        log_file_handler.setFormatter(formatter)
        root.addHandler(log_file_handler)


def setup_output_dir(base_output_dir, continue_training_from) -> str:
    if continue_training_from is None:
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_dir = start_time
    else:
        training_dir = continue_training_from

    output_dir = f'{base_output_dir}/{training_dir}/'
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def run_training(params: Parameters):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    output_dir = setup_output_dir(params.output_dir, params.continue_training_from)
    setup_logging(output_dir)

    logging.info(f"All the output artifacts would go to {output_dir}")
    logging.info(f"Torch device is : {params.device}")

    logging.info("Running training with the following parameters: ")
    logging.info(params)

    dm = load_dataset(params.dataset)

    if params.random_train_assets is not None:
        assets_num = params.random_train_assets
    else:
        assets_num = len(dm.global_matrix.asset)

    # model_dim = assets_num
    # model_dim = len(dm.features)
    model_dim = params.model_dim

    d_encoder_decoder_embedding = params.d_encoder_decoder_embedding
    if d_encoder_decoder_embedding is None:
        d_encoder_decoder_embedding = params.multihead_num * model_dim

    if params.continue_training_from is not None:
        model = torch.load(output_dir + '/best_model.pkl', map_location=params.device)
    else:
        model = make_model(params.dataset.batch_size, assets_num, params.dataset.x_window_size, dm.features,
                           N=params.encoder_decoder_layers, d_model_Encoder=d_encoder_decoder_embedding,
                           d_model_Decoder=d_encoder_decoder_embedding,
                           d_ff_Encoder=d_encoder_decoder_embedding,
                           d_ff_Decoder=d_encoder_decoder_embedding,
                           h=params.multihead_num,
                           dropout=0.01,
                           local_context_length=params.local_context_length,
                           activation=params.activation)

        model = model.to(params.device)

    ##########################train net####################################################
    tst_loss, tst_portfolio_value = train_net(dm, params.total_step, params.output_step,
                                              params.dataset.batch_size,
                                              params.dataset.x_window_size,
                                              params.local_context_length, model,
                                              output_dir,
                                              params.device,
                                              params.learning_rate, params.weight_decay,
                                              params.interest_rate,
                                              params.trading_consumption, params.variance_penalty,
                                              params.cost_penalty,
                                              params.random_train_assets,
                                              params.test_dataset_assets,
                                              params.save_last_model,
                                              params.continue_training_from,
                                              params.gradient_clipping_max,
                                              params.custom_commission_loss,
                                              params.validation_online,
                                              params.save_model_steps)

    return model, tst_loss, tst_portfolio_value


def run_test(run_dir, params: Parameters, test_type='validation', test_model='best_model'):
    assert test_type in ['train', 'validation', 'test']

    setup_logging()

    dm = load_dataset(params.dataset)

    logging.info(f"Evaluating the {test_type} dataset...")

    device = "cpu"
    model = torch.load(run_dir + f'/{test_model}.pkl', map_location=device)
    model = model.to(device)

    if test_type == 'test':
        test_indices = dm._test_ind
    elif test_type == 'validation':
        test_indices = dm._validation_ind
    elif test_type == 'train':
        test_indices = dm._train_ind
    else:
        raise NotImplementedError()

    ##########################test net#####################################################
    portfolio_value_history, rewards, SR, CR, tst_pc_array, TO, tst_long_term_w, tst_trg_y = test_net(
        dm, test_indices, params.dataset.x_window_size, params.local_context_length, model, device,
        params.trading_consumption, params.interest_rate, params.variance_penalty, params.cost_penalty,
        params.test_dataset_assets, params.custom_commission_loss)

    test_assets = params.test_dataset_assets
    asset_names = pd.Series([asset.lower() for asset in test_assets])
    test_period_time = dm.global_matrix.time[
                       test_indices[0] + params.dataset.x_window_size:test_indices[-1] + 1].to_series().tolist()

    portfolio_distribution = pd.DataFrame(
        tst_long_term_w.reshape(-1, len(test_assets) + 1),
        columns=['asset_cash'] + ('asset_' + asset_names).tolist(),
        index=test_period_time)

    price_change_df = pd.DataFrame(tst_trg_y[..., 0].reshape(-1, len(test_assets)),
                                   columns='price_change_' + asset_names,
                                   index=test_period_time)

    test_results = portfolio_distribution.assign(rewards=rewards,
                                                 portfolio_value=portfolio_value_history)

    test_results = test_results.join(price_change_df)
    test_results.to_csv(f"{run_dir}/test_results.csv")

    csv_dir = f"{run_dir}/test_summary.csv"
    d = {"fAPV": [portfolio_value_history[-1].item()],
         "SR": [SR.item()],
         "CR": [CR.item()],
         "TO": [TO.item()],
         "St_v": [''.join(str(e) + ', ' for e in portfolio_value_history)],
         "backtest_test_history": [''.join(str(e) + ', ' for e in tst_pc_array.cpu().numpy())],
         }
    dataframe = pd.DataFrame(data=d)
    dataframe.to_csv(csv_dir)
