import netCDF4 as nc
import numpy as np
import scipy.stats as st
import xarray as xr

import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# Required for feeding the data iinto NN.
class myDataset(Dataset):
    def __init__(self, X, Y):
        self.features = torch.tensor(X, dtype=torch.float64)
        self.labels = torch.tensor(Y, dtype=torch.float64)

    def __len__(self):
        return len(self.features.T)

    def __getitem__(self, idx):
        feature = self.features[:, idx]
        label = self.labels[:, idx]

        return feature, label


# The NN model.
class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(564, 5000, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(5000, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 500, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(500, 140, dtype=torch.float64),
        )

    def forward(self, X):
        return self.linear_stack(X)


# training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    avg_loss = 0
    for batch, (X, Y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            avg_loss += loss.item()

    avg_loss /= len(dataloader)

    return avg_loss


# validating loop
def val_loop(dataloader, model, loss_fn):
    avg_loss = 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, Y)
            avg_loss += loss.item()

    avg_loss /= len(dataloader)

    return avg_loss
