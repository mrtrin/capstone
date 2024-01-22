from .datasets import load_train_test
from .models import airmodel
from sklearn.model_selection import train_test_split

import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

def main():
    X, y = load_train_test()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = airmodel.AirModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

if __name__ == '__main__':
    main()