import torch, sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import TQDMProgressBar


class JoystickPressureModel(pl.LightningModule):
    def __init__(self, hidden_size: int = 16, num_layers: int = 1, learning_rate=0.003):
        super(JoystickPressureModel, self).__init__()

        self.lr = learning_rate
        self.__model = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.mse_loss_fc = nn.MSELoss(reduction='mean')
        self.mae_loss_fc = nn.L1Loss(reduction='mean')

        self.__test_data = list()
        self.__test_pred = list()

    def forward(self, input_data):
        out, _ = self.__model(input_data)
        out = self.fc(out[:, -1])

        return out.squeeze()

    def training_step(self, batch, batch_idx):
        data, label = batch

        #print(data.shape, label.shape)
        pred = self.forward(data)
        mse_loss = self.mse_loss_fc(pred, label)

        self.log(name='train_loss', value=mse_loss, prog_bar=True)
        self.log(name='pressure_loss', value=self.mae_loss_fc(pred, label), prog_bar=False)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)
        mse_loss = self.mse_loss_fc(pred, label)

        self.log(name='val_loss', value=mse_loss, prog_bar=True)
        self.log(name='pressure_loss', value=self.mae_loss_fc(pred, label), prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)
        mse_loss = self.mse_loss_fc(pred, label)

        self.log(name='test_loss', value=mse_loss, prog_bar=True)
        self.log(name='pressure_loss', value=self.mae_loss_fc(pred, label), prog_bar=True)

        data_arr = data.cpu().numpy()
        test_data = data_arr[0, 0, :]

        pred_arr = pred.cpu().numpy()

        self.__test_data.append(test_data)
        t_last = data_arr[0, -1, 0]

        self.__test_data.append(test_data)
        #self.__test_pred.append(np.vstack((t_last, pred_arr)).T)

        self.__test_pred.append(np.array([t_last, pred_arr[-1]]))

    def on_test_epoch_end(self):
        df = pd.DataFrame(np.vstack(self.__test_data), columns=['time', 'pressure_measurement'])
        df.to_csv(path_or_buf='test_data.csv', index=False)

        df = pd.DataFrame(np.vstack(self.__test_pred), columns=['time', 'pressure_prediction'])
        df.to_csv(path_or_buf='test_pred.csv', index=False)

        print("Test results saved to test_results.csv")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
