import torch, sys, torchmetrics
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar


class JoystickPressureModel(pl.LightningModule):
    def __init__(self, hidden_size=256, num_layers=2, learning_rate=0.003):
        super(JoystickPressureModel, self).__init__()

        self.lr = learning_rate
        self.__seq_len = 2800
        self.__model = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.criterion = nn.MSELoss()

    def forward(self, input_data):
        out, _ = self.__model(input_data)

        return out.squeeze()

    def training_step(self, batch, batch_idx):
        data, labels = batch
        pred = self.forward(data)
        loss = self.criterion(pred, labels)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        pred = self.forward(data)
        loss = self.criterion(pred, labels)
        #acc = (predictions.argmax(dim=1) == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        #self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
