import logging

import numpy as np
import pandas as pd
import xarray as xr
import torch
from sparsemax import sparsemax, Sparsemax

from pgportfolio.marketdata.replaybuffer import ReplayBuffer
from pgportfolio.tools.configprocess import parse_time
from pgportfolio.tools.data import panel_fillna
from pgportfolio.tools.indicator import calculate_macd, calculate_signal_line, calculate_rsi


class DataMatrices:
    def __init__(self, dataset_file,
                 dataset_assets=None,
                 dataset_date_range=None,
                 dataset_features=None,
                 batch_size=50,
                 buffer_bias_ratio=0,
                 window_size=50,
                 validation_portion=0.15,
                 test_portion=0.15,
                 train_range=None,
                 validation_range=None,
                 test_range=None,
                 assets_per_batch=None,
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

        self.full_data_matrix = da

        if dataset_assets is not None:
            da = da.sel(asset=dataset_assets)

        indicator_features = ['rsi', 'macd', 'signal_line']

        if dataset_features:
            non_indicator_features = [feature for feature in dataset_features if feature not in indicator_features]
            da = da.sel(feature=non_indicator_features)
            used_features = dataset_features
            indicator_features = list(filter(lambda feature: feature in used_features, indicator_features))
        else:
            used_features = list(da.feature.values) + indicator_features

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

        da = panel_fillna(da, "ffill")

        # Cut off the first 300 days and all assets that have nans or infs after that period
        cut_off_days = 300
        days_with_nans_per_asset_and_feature = (np.isnan(da) | np.isinf(da)).sum(axis=(2))
        days_with_nans_per_asset = np.max(days_with_nans_per_asset_and_feature, axis=0)
        assets_with_inf_values = da.where(np.isinf(da), drop=True).asset.values

        good_assets = days_with_nans_per_asset[days_with_nans_per_asset <= cut_off_days].asset.values
        good_assets = [a for a in good_assets if a not in assets_with_inf_values]

        if 'pe_ratio_ttm' in da.feature.values:
            std = da.sel(feature='pe_ratio_ttm', asset=good_assets).std(axis=1)
            assets_with_large_pe_std = std[std > 100].asset.values
            good_assets = [a for a in good_assets if a not in assets_with_large_pe_std]

        da = da.sel(asset=good_assets)
        da = da.where(da.time >= da.time[cut_off_days], drop=True)

        self.bad_assets = sorted(da.where(np.isnan(da) | np.isinf(da), drop=True).asset.values)

        self.good_assets_indices = np.argwhere(np.isin(da.asset.values, good_assets)).squeeze(1)

        logging.info(f"Found bad values for the following {len(self.bad_assets)} assets: {self.bad_assets}")

        self.__global_data = da
        self.__global_data_partially_norm = self.__global_data.copy()

        normalize_features = [f for f in ['pe_ratio_ttm', 'eps_surprise_percentage', 'rsi'] if
                              f in used_features]
        if len(normalize_features) > 0:
            self.__global_data_partially_norm.loc[normalize_features, ...] = \
                self.__global_data_partially_norm.loc[normalize_features, ...] / 100

        # assert window_size >= MIN_NUM_PERIOD
        self.__coin_no = len(self.__global_data.asset)
        self.assets = self.__global_data.asset.values
        self.features = self.__global_data.feature.values

        # portfolio vector memory, [time, assets]
        self.__PVM = pd.DataFrame(index=self.__global_data.time,
                                  columns=self.__global_data.asset)
        if assets_per_batch:
            num_assets = assets_per_batch
        else:
            num_assets = self.__coin_no

        self.__PVM = self.__PVM.fillna(1.0 / num_assets)
        # self.__PVM = self.__PVM.fillna(0)
        # sparsemax = Sparsemax(-1)
        # self.__PVM.loc[:,:] = sparsemax(torch.tensor(np.random.random(self.__PVM.shape))).detach().numpy()
        # self.__PVM = self.__PVM.astype('float')

        self._window_size = window_size
        self._num_periods = len(self.__global_data.time)
        if test_portion is not None and validation_portion is not None:
            self.__divide_data_by_ratio(validation_portion, test_portion, portion_reversed)
        elif train_range is not None and validation_range is not None and test_range is not None:
            self.__divide_data_by_dates(train_range, validation_range, test_range)
        else:
            raise ValueError("Invalid train/validation/test periods provided")

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

    def __get_assets_indices(self, assets):
        if assets is None:
            assets_indices = slice(None)
        else:
            assets_indices = pd.Series(range(len(self.global_matrix.asset)), index=self.global_matrix.asset.values) \
                [assets].to_numpy()

        return assets_indices

    def get_test_set(self, assets=None, normalized=False, get_last_w_omega=False):
        assets_indices = self.__get_assets_indices(assets)
        return self.__pack_samples(assets_indices, self.test_indices, get_last_w_omega, normalized=normalized)

    def get_validation_set(self, assets=None, normalized=False, get_last_w_omega=False):
        assets_indices = self.__get_assets_indices(assets)
        return self.__pack_samples(assets_indices, self.validation_indices, get_last_w_omega, normalized=normalized)

    def get_test_set_online(self, assets, ind_start, ind_end, x_window_size, get_last_w_omega):
        assets_indices = self.__get_assets_indices(assets)
        return self.__pack_samples_test_online(assets_indices, ind_start, ind_end, x_window_size, get_last_w_omega)

    def get_training_set(self, assets, get_last_w_omega=False, normalized=False):
        assets_indices = self.__get_assets_indices(assets)
        return self.__pack_samples(assets_indices, self._train_ind[:-self._window_size], get_last_w_omega, normalized=normalized)

    ##############################################################################
    def next_batch(self, custom_commission_loss, random_assets=None):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """

        while True:
            if random_assets is None:
                assets_indices = slice(None)
                apply_softmax_to_weights = False
            else:
                # assets_indices = np.random.choice(range(len(self.global_matrix.asset)), random_assets, replace=False)
                assets_indices = np.random.choice(self.good_assets_indices, random_assets, replace=False)
                apply_softmax_to_weights = True

            experience_indices = [exp.state_index for exp in self.__replay_buffer.next_experience_batch()]
            batch = self.__pack_samples(assets_indices, experience_indices,
                                        custom_commission_loss,
                                        normalized=True)

            # if apply_softmax_to_weights:
            #   batch['last_w'] = torch.nn.functional.softmax(torch.tensor(batch['last_w']), dim=1).numpy()
            batch['last_w'] = torch.nn.functional.softmax(torch.tensor(np.random.random((batch['last_w'].shape[0], batch['last_w'].shape[1]+1))), dim=1).numpy()[:, 1:]

            nans = np.isnan(batch['X'])
            infs = np.isinf(batch['X'])
            if random_assets is None or (nans.sum() == 0 and infs.sum() == 0):
                batch['assets_indices'] = assets_indices
                batch['experience_indices'] = experience_indices
                break
            assets_with_null = assets_indices[np.argwhere(nans.sum(axis=(0, 1, 3)) > 0)]
            assets_with_inf = assets_indices[np.argwhere(infs.sum(axis=(0, 1, 3)) > 0)]
            logging.info(f"Found null features for assets {assets_with_null} "
                         f"and inf features for assets {assets_with_inf} in experiences {experience_indices}")

        return batch

    def __pack_samples(self, assets_indices, indexs, get_last_w_omega, normalized=False):
        indexs = np.array(indexs)

        last_w = self.__PVM.iloc[indexs - 1, assets_indices].values

        def setw(w):
            self.__PVM.iloc[indexs, assets_indices] = w

        #            logging.info("set w index from %d-%d!" %( indexs[0],indexs[-1]))
        M = [self.get_submatrix(assets_indices, index, index + self._window_size + 1, normalized=normalized) for index
             in
             indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]

        if get_last_w_omega:
            y_last = M[:, 0, :, -2] / M[:, 0, :, -3]
            y_last_with_cash = np.concatenate([np.ones((y_last.shape[0], 1)), y_last], -1) # [128,13]

            last_w_with_cash = np.concatenate([(1 - last_w.sum(axis=-1, keepdims=True)), last_w], -1) #[128, 13]
            last_w_omega = last_w * y_last \
                           /   np.expand_dims((last_w_with_cash*y_last_with_cash).sum(-1), -1) # [128,1,12]
            last_w = last_w_omega

        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    def __pack_samples_test_online(self, assets_indices, ind_start, ind_end, x_window_size, get_last_w_omega):
        last_w = self.__PVM.values[ind_start - 1:ind_start, assets_indices]

        #        y_window_size = window_size-x_window_size
        def setw(w):
            self.__PVM.iloc[ind_start, assets_indices] = w

        #            logging.info("set w index from %d-%d!" %( indexs[0],indexs[-1]))
        M = [self.get_submatrix(assets_indices, ind_start, ind_end, normalized=True)]  # [1,4,11,2807]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, x_window_size:] / M[:, 0, None, :, x_window_size - 1:-1]

        if get_last_w_omega:
            y_last = M[:, 0, :, x_window_size-1] / M[:, 0, :, x_window_size-2]
            y_last_with_cash = np.concatenate([np.ones((y_last.shape[0], 1)), y_last], -1) # [128,13]

            last_w_with_cash = np.concatenate([(1 - last_w.sum(axis=-1, keepdims=True)), last_w], -1) #[128, 13]
            last_w_omega = last_w * y_last \
                           /   np.expand_dims((last_w_with_cash*y_last_with_cash).sum(-1), -1) # [128,1,12]
            last_w = last_w_omega

        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    ##############################################################################################
    def get_submatrix(self, assets_indices, ind_start, ind_end, normalized=False):
        if normalized:
            src = self.__global_data_partially_norm
        else:
            src = self.__global_data

        return src.values[:, assets_indices, ind_start:ind_end]

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

    def __get_time_indices_by_date_range(self, date_range):
        if date_range[0] is None:
            start = 0
        else:
            start = np.argwhere(self.global_matrix.time.values >= np.datetime64(date_range[0]))[0].item()

        if date_range[1] is None:
            end = len(self.global_matrix.time)
        else:
            end = np.argwhere(self.global_matrix.time.values < np.datetime64(date_range[1]))[-1].item() + 1

        return np.arange(start, end)

    def __divide_data_by_dates(self, train_range, validation_range, test_range):
        self._train_ind = self.__get_time_indices_by_date_range(train_range)
        self._validation_ind = self.__get_time_indices_by_date_range(validation_range)
        self._test_ind = self.__get_time_indices_by_date_range(test_range)

        self._num_train_samples = len(self._train_ind)
        self._num_validation_samples = len(self.validation_indices)
        self._num_test_samples = len(self.test_indices)
