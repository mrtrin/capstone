from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from src.datasets import load_train_test
from src.portfolio import Portfolio

import pandas as pd

def main(num_epochs=1, batch_size=32, learning_rate=0.001, shuffle=False):
    yfdata, features, targets, X, y, y_price = load_train_test()
    
    print('========= Original Dataset =========')
    print('features', features.shape, 'columns', features.index[0], features.index[-1])
    print('targets', targets.shape, 'columns', targets.index[0], targets.index[-1])
    print('X', X.shape)
    print('y', y.shape)
    print('y_price', y_price.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('========= Train / Test Dataset =========')
    print('X_train', X_train.shape)
    print('X_test', X_test.shape)
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)
    print('y_price', y_price.shape)

    # TODO: fix model so loss are reduced
    model = train(X_train, y_train, num_epochs, batch_size, learning_rate, shuffle)

    # # TODO: show Portfolio value from train dataset
    optimal_test_portfolio = Portfolio.portfolio_returns(y, y_price)
    print(optimal_test_portfolio)
    print((optimal_test_portfolio+1).cumprod())
    predicted_test_portfolio = Portfolio.portfolio_returns(model.predict(X_test), y_price)

    print('----------- predicted -----------')
    print(predicted_test_portfolio)
    print((predicted_test_portfolio+1).cumprod())

    # TODO: show Portfolio value from test dataset


def train(X_train, y_train, num_epochs, batch_size, learning_rate, shuffle):

    model = Sequential()
    model.add(LSTM(units=X_train.shape[2]*4, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=y_train.shape[1]))  # output for optimal weights for next week.

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return model

if __name__ == '__main__':
    main()
