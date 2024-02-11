from sklearn.model_selection import train_test_split
from src.datasets import load_train_test
from src.models import airmodel

from statsmodels.graphics.gofplots import qqplot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize


def main(num_epochs=1000, batch_size=128, learning_rate=0.001, shuffle=False):
    # TODO: More Feature Engineering, Normalization
    yfdata, features, targets, X, y = load_train_test()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('========= Train / Test Dataset =========')
    print('X_train', X_train.shape)
    print('X_test', X_test.shape)
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)

    # TODO: fix model so loss are reduced
    model = train(X_train, y_train, 1000, batch_size, learning_rate, shuffle)

    # TODO: show Portfolio value from train dataset

    # TODO: show Portfolio value from test dataset


def train(X_train, y_train, num_epochs, batch_size, learning_rate, shuffle):
    
    # Setup Model and Optimizer
    model = airmodel.AirModel(X_train.shape[2], y_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    train_loader = data.DataLoader(data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=shuffle)

    model.train()

    for epoch in range(num_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            print(x.shape)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        model.eval()
        print(f'epoch {epoch}/{num_epochs}, loss: {loss.item():.4f}')
    
    return model
    

if __name__ == '__main__':
    main()
