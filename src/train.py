from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from src.datasets import load_train_test
from src.portfolio import Portfolio

import pandas as pd

def main(num_epochs=1, batch_size=32, learning_rate=0.001, shuffle=False):
    yfdata, features, targets, X, y, y_price = load_train_test()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('========= Train / Test Dataset =========')
    print('X_train', X_train.shape)
    print('X_test', X_test.shape)
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)
    print('y_price', y_price.shape)
    print('X columns', features.columns)
    print('Y columns', targets.columns)

    # TODO: fix model so loss are reduced
    # model = train(X_train, y_train, num_epochs, batch_size, learning_rate, shuffle)

    # # TODO: show Portfolio value from train dataset
    optimal_test_portfolio = Portfolio.compute_portfolio(y, y_price)
    # predicted_test_portfolio = Portfolio.compute_portfolio(model.predict(X_test), )

    # TODO: show Portfolio value from test dataset


def train(X_train, y_train, num_epochs, batch_size, learning_rate, shuffle):
    model = Sequential()
    model.add(LSTM(units=X_train.shape[2]*4, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=y_train.shape[1]))  # output for optimal weights for next week.

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return model

if __name__ == '__main__':
    main()
