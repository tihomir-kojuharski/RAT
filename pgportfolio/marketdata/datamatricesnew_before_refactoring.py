import logging

import numpy as np
import pandas as pd
import xarray as xr

from pgportfolio.marketdata.replaybuffer import ReplayBuffer
from pgportfolio.tools.configprocess import parse_time
from pgportfolio.tools.data import panel_fillna
from pgportfolio.tools.indicator import calculate_macd, calculate_signal_line, calculate_rsi


class DataMatricesNew:
    def __init__(self, dataset_file,
                 dataset_assets=None,
                 dataset_date_range=None,
                 dataset_features=None,
                 batch_size=50,
                 buffer_bias_ratio=0,
                 window_size=50,
                 validation_portion=0.15,
                 test_portion=0.15,
                 train_date=None,
                 validation_date=None,
                 test_date=None,
                 portion_reversed=False, is_permed=False):
        """
        :param window_size: periods of input data
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param validation_portion: portion of test set
        :param test_portion: portion of test set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        """

        da = xr.open_dataarray(dataset_file)

        if dataset_assets is not None:
            da = da.sel(asset=dataset_assets)

        indicator_features = ['rsi', 'macd', 'signal_line']

        if dataset_features:
            da = da.sel(feature=dataset_features)
            used_features = dataset_features
            indicator_features = list(filter(lambda feature: feature in used_features, indicator_features))
        else:
            used_features = list(da.asset.values) + indicator_features

        if len(indicator_features) > 0:
            da = xr.concat([da, xr.DataArray(coords=[indicator_features], dims=['feature'])], dim='feature')

            for asset in da.asset.values:
                close_price = da.sel(feature='close', asset=asset).to_series().dropna()

                if 'macd' in used_features or 'signal_line' in used_features:
                    macd = calculate_macd(close_price)
                    if 'macd' in used_features:
                        da.loc['macd', asset, close_price.index] = macd

                    if 'signal_line' in used_features:
                        signal_line = calculate_signal_line(macd)
                        da.loc['signal_line', asset, close_price.index] = signal_line

                if 'rsi' in used_features:
                    rsi = calculate_rsi(close_price, 14, asset)
                    da.loc['rsi', asset, close_price.index] = rsi


        if dataset_date_range:
            da = da.sel(time=dataset_date_range)

        self.__global_data = panel_fillna(da, "both")

        # assert window_size >= MIN_NUM_PERIOD
        self.__coin_no = len(self.__global_data.asset)
        self.features = len(self.__global_data.feature)

        # portfolio vector memory, [time, assets]
        self.__PVM = pd.DataFrame(index=self.__global_data.time,
                                  columns=self.__global_data.asset)
        self.__PVM = self.__PVM.fillna(1.0 / self.__coin_no)

        self._window_size = window_size
        self._num_periods = len(self.__global_data.time)
        if test_portion is not None and validation_portion is not None:
            self.__divide_data_by_ratio(validation_portion, test_portion, portion_reversed)
        elif test_date is not None and validation_date is not None:
            self.__divide_data_by_dates(validation_date, test_date, train_date)

        self._portion_reversed = portion_reversed
        self.__is_permed = is_permed

        self.__batch_size = batch_size
        self.__delta = 0  # the count of global increased
        end_index = self._train_ind[-1]
        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                            end_index=end_index,
                                            sample_bias=buffer_bias_ratio,
                                            batch_size=self.__batch_size,
                                            coin_number=self.__coin_no,
                                            is_permed=self.__is_permed)

        logging.info("the number of training/validation/test examples is %s/%s/%s"
                     % (self._num_train_samples, self._num_validation_samples, self._num_test_samples))

        datasets = [('training', self._train_ind), ('validation', self._validation_ind), ('test', self._test_ind)]
        for dataset_name, indices in datasets:
            min_index = min(indices)
            max_index = max(indices)

            min_date = self.global_matrix.time[min_index].dt.strftime('%Y-%m-%d').item()
            max_date = self.global_matrix.time[max_index].dt.strftime('%Y-%m-%d').item()

            logging.info(f"the {dataset_name} set is from {min_index} to {max_index} (from {min_date} to {max_date})")

    @property
    def global_weights(self):
        return self.__PVM

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def test_indices(self):
        return self._test_ind[:-(self._window_size + 1):]

    @property
    def validation_indices(self):
        return self._validation_ind[:-(self._window_size + 1):]

    @property
    def num_test_samples(self):
        return self._num_test_samples

    def append_experience(self, online_w=None):
        """
        :param online_w: (number of assets + 1, ) numpy array
        Let it be None if in the backtest case.
        """
        self.__delta += 1
        self._train_ind.append(self._train_ind[-1] + 1)
        appended_index = self._train_ind[-1]
        self.__replay_buffer.append_experience(appended_index)

    def get_test_set(self):
        return self.__pack_samples(self.test_indices)

    def get_validation_set(self):
        return self.__pack_samples(self.validation_indices)

    def get_test_set_online(self, ind_start, ind_end, x_window_size):
        return self.__pack_samples_test_online(ind_start, ind_end, x_window_size)

    def get_training_set(self):
        return self.__pack_samples(self._train_ind[:-self._window_size])

    ##############################################################################
    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        #        logging.info(np.shape([exp.state_index for exp in self.__replay_buffer.next_experience_batch()]),[exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs - 1, :]

        def setw(w):
            self.__PVM.iloc[indexs, :] = w

        #            logging.info("set w index from %d-%d!" %( indexs[0],indexs[-1]))
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    def __pack_samples_test_online(self, ind_start, ind_end, x_window_size):
        #        indexs = np.array(indexs)
        last_w = self.__PVM.values[ind_start - 1:ind_start, :]

        #        y_window_size = window_size-x_window_size
        def setw(w):
            self.__PVM.iloc[ind_start, :] = w

        #            logging.info("set w index from %d-%d!" %( indexs[0],indexs[-1]))
        M = [self.get_submatrix_test_online(ind_start, ind_end)]  # [1,4,11,2807]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, x_window_size:] / M[:, 0, None, :, x_window_size - 1:-1]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    ##############################################################################################
    def get_submatrix(self, ind):
        return self.__global_data.values[:, :, ind:ind + self._window_size + 1]

    def get_submatrix_test_online(self, ind_start, ind_end):
        return self.__global_data.values[:, :, ind_start:ind_end]

    def __divide_data_by_ratio(self, validation_portion, test_portion, portion_reversed):
        train_portion = 1 - (validation_portion + test_portion)
        s = float(train_portion + validation_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion, validation_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            portion_split[1] = portion_split[0] + portion_split[1]
            indices = np.arange(self._num_periods)
            self._test_ind, self._validation_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion, validation_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            portion_split[1] = portion_split[0] + portion_split[1]
            indices = np.arange(self._num_periods)
            self._train_ind, self._validation_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[:-(self._window_size + 1)]
        # NOTE(zhengyao): change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_validation_samples = len(self.validation_indices)
        self._num_test_samples = len(self.test_indices)

    def __divide_data_by_dates(self, validation_date, test_date, train_date):
        test_len = len(self.global_matrix.sel(time=slice(test_date, None)).time)
        validation_len = len(self.global_matrix.sel(time=slice(validation_date, test_date)).time)
        total_len = len(self.global_matrix.time)

        if train_date is not None:
            train_len = len(self.global_matrix.sel(time=slice(train_date, validation_date)).time)
        else:
            train_len = total_len - validation_len - test_len

        train_start_idx = total_len - test_len - validation_len - train_len
        self._train_ind = np.arange(train_start_idx, train_start_idx + train_len)

        validation_start_idx = self._train_ind[-1] + 1
        self._validation_ind = np.arange(validation_start_idx, validation_start_idx + validation_len)

        test_start_idx = self._validation_ind[-1] + 1
        self._test_ind = np.arange(test_start_idx, test_start_idx + test_len)

        self._num_train_samples = len(self._train_ind)
        self._num_validation_samples = len(self.validation_indices)
        self._num_test_samples = len(self.test_indices)
