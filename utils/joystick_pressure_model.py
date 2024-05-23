import torch, sys
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar


class JoystickPressureModel(pl.LightningModule):
    def __init__(self, hidden_size: int = 16, num_layers: int = 1, learning_rate=0.003):
        super(JoystickPressureModel, self).__init__()

        self.lr = learning_rate
        self.__model = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.mse_loss_fc = nn.MSELoss(reduction='mean')
        self.mae_loss_fc = nn.L1Loss(reduction='mean')

    def forward(self, input_data):
        out, _ = self.__model(input_data)
        out = self.fc(out[:, -1])

        return out.squeeze()

    def training_step(self, batch, batch_idx):
        data, label = batch
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

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
