import pickle, torch

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset


class JoystickDataset(Dataset):
    def __init__(self, joystick_data: np.array, seq_len: int = 1, forecast_distance: int = 1):
        self.__seq_len = seq_len
        self.__forecast__dist = forecast_distance
        self.__dataset_tensor = torch.FloatTensor(joystick_data)

    def __len__(self):
        return self.__dataset_tensor.shape[0] - (self.__seq_len + self.__forecast__dist + 1)

    def __getitem__(self, idx):
        return (self.__dataset_tensor[idx: idx+self.__seq_len],
                self.__dataset_tensor[idx + self.__seq_len + self.__forecast__dist, 1])


class JoystickDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, seq_len: int = 1, forecast_distance: int = 1, batch_size: int = 32,
                 n_of_worker: int = 1):
        super().__init__()

        self.__dataset_path = dataset_path
        self.__batch_size = batch_size
        self.__n_of_worker = n_of_worker
        self.__seq_len = seq_len
        self.__forecast_dist = forecast_distance
        #self.__dataset: JoystickDataset

        with open(dataset_path, 'rb') as f:
            self.__peak_dataset = pickle.load(f)

        self.__raw_data_1_df = self.__peak_dataset['raw_data_1'].dropna()
        self.__raw_data_2_df = self.__peak_dataset['raw_data_2'].dropna()
        self.__raw_data_3_df = self.__peak_dataset['raw_data_3'].dropna()
        self.__raw_data_4_df = self.__peak_dataset['raw_data_4'].dropna()
        self.__raw_data_5_df = self.__peak_dataset['raw_data_5'].dropna()

        arr_list = list()

        for i in range(4):
            arr_list.append(self.__raw_data_1_df.iloc[:, i * 2:i * 2 + 2].to_numpy())

        self.__raw_data_1_arr = np.concatenate(arr_list)

        arr_list.clear()

        for i in range(4):
            arr_list.append(self.__raw_data_2_df.iloc[:, i * 2:i * 2 + 2].to_numpy())

        self.__raw_data_2_arr = np.concatenate(arr_list)

        arr_list.clear()

        for i in range(4):
            arr_list.append(self.__raw_data_3_df.iloc[:, i * 2:i * 2 + 2].to_numpy())

        self.__raw_data_3_arr = np.concatenate(arr_list)

        arr_list.clear()

        for i in range(4):
            arr_list.append(self.__raw_data_4_df.iloc[:, i * 2:i * 2 + 2].to_numpy())

        self.__raw_data_4_arr = np.concatenate(arr_list)

        arr_list.clear()

        for i in range(4):
            arr_list.append(self.__raw_data_5_df.iloc[:, i * 2:i * 2 + 2].to_numpy())

        self.__raw_data_5_arr = np.concatenate(arr_list)

        arr_list.clear()

        self.__raw_data_arr = np.concatenate((self.__raw_data_1_arr, self.__raw_data_2_arr, self.__raw_data_3_arr,
                                              self.__raw_data_4_arr), axis=0)
        #self.__raw_data_1_arr = self.__raw_data_5_arr[0:int(self.__raw_data_1_arr[0]/4)]

    def setup(self, stage=None):
        self.__dataset = JoystickDataset(joystick_data=self.__raw_data_arr,
                                         seq_len=self.__seq_len, forecast_distance=self.__forecast_dist)

        self.__train_dataset, self.__val_dataset = torch.utils.data.random_split(dataset=self.__dataset,
                                                                                 lengths=[0.8, 0.2])

        self.__test_dataset = JoystickDataset(joystick_data=self.__raw_data_5_arr,
                                              seq_len=self.__seq_len, forecast_distance=self.__forecast_dist)

    def train_dataloader(self):
        return DataLoader(dataset=self.__train_dataset, batch_size=self.__batch_size, shuffle=True, num_workers=self.__n_of_worker,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.__val_dataset, batch_size=self.__batch_size, shuffle=False, num_workers=self.__n_of_worker,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.__test_dataset, batch_size=self.__batch_size, shuffle=False, num_workers=self.__n_of_worker,
                          persistent_workers=True)
