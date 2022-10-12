import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.core.lightning import LightningModule

import sys

sys.path.append('../../')
from pmlayer.torch import hierarchical_lattice_layer


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float)
        self.y = torch.tensor(y, dtype = torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class PartiallyMonotoneDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        # generate data
        num_data_train = 1000
        num_data_val = 100
        num_data_test = 100
        len_feature_vector = 2
        self.x_train = np.random.rand(num_data_train, len_feature_vector)
        self.x_val = np.random.rand(num_data_val, len_feature_vector)
        self.x_test = np.random.rand(num_data_test, len_feature_vector)
        self.y_train = self.__square(self.x_train).reshape(-1)
        self.y_val = self.__square(self.x_val).reshape(-1)
        self.y_test = self.__square(self.x_test).reshape(-1)

    def __square(self, x):
        return (x[:,0] * x[:,0] + x[:,1] * x[:,1]) / 2.0

    def train_dataloader(self):
        td = TorchDataset(self.x_train, self.y_train)
        return torch.utils.data.DataLoader(td,
                                           batch_size=128,
                                           shuffle=True)

    def val_dataloader(self):
        td = TorchDataset(self.x_val, self.y_val)
        return torch.utils.data.DataLoader(td,
                                           batch_size=99999999,
                                           shuffle=False)

    def test_dataloader(self):
        td = TorchDataset(self.x_test, self.y_test)
        return torch.utils.data.DataLoader(td,
                                           batch_size=99999999,
                                           shuffle=False)

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = hierarchical_lattice_layer.HLattice([4,4],[1,1])
        self.loss_function = nn.MSELoss(reduction='mean')

    def forward(self, x, lengths=None):
        return self.model(x).view(-1)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        return loss

if __name__ == '__main__':
    pmdm = PartiallyMonotoneDataModule()

    # train
    model = Model()
    trainer = pl.Trainer(max_epochs=100)
    train_dataloader = pmdm.train_dataloader()
    val_dataloader = pmdm.val_dataloader()
    trainer.fit(model, train_dataloader, val_dataloader)

    # test
    test_dataloader = pmdm.test_dataloader()
    trainer.test(dataloaders=test_dataloader)
