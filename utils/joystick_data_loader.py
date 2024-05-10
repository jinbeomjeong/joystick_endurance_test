import pickle, torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset


class JoystickDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.__peak_dataset = dict()
        #self.__peak_data_1 = pd.DataFrame()

        with open(dataset_path, 'rb') as f:
            self.__peak_dataset = pickle.load(f)

        self.__peak_data = self.__peak_dataset['raw_data_1'].dropna()
        self.__input_data = self.__peak_data[['time_p1(hour)', 'port_1_p(bar)']].to_numpy()

        self.__output_data = self.__peak_data['port_1_p(bar)'].to_numpy()

    def __len__(self):
        return self.__input_data.shape[0]

    def __getitem__(self, idx):
        self.__input_data = torch.FloatTensor(self.__input_data)
        self.__output_data = torch.FloatTensor(self.__output_data)

        return self.__input_data[idx], self.__output_data[idx]


class JoystickDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size=32, n_of_worker=1):
        super().__init__()

        self.__dataset_path = dataset_path
        self.__batch_size = batch_size
        self.__n_of_worker = n_of_worker

    def setup(self, stage=None):
        self.__dataset = JoystickDataset(dataset_path=self.__dataset_path)

    def train_dataloader(self):
        return DataLoader(dataset=self.__dataset, batch_size=self.__batch_size, shuffle=True, num_workers=self.__n_of_worker,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.__dataset, batch_size=self.__batch_size, shuffle=False, num_workers=self.__n_of_worker,
                          persistent_workers=True)
