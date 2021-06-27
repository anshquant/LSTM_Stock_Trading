# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:08:33 2021

@author: Anshumaan Gandhi
"""
import pandas as pd
import pandas_datareader as dr
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from AlphaVantage_data import get_intradaydata
from LSTM_Daily_Backtest import backtest_daily_swing




def download_data_daily(ticker, start_date, end_date):
    
    df = dr.data.get_data_yahoo(ticker, start_date, end_date)
    df['Returns'] = df.Close.pct_change()
    df['Log_returns'] = np.log(1 + df['Returns'])
    df.dropna(inplace=True)
    X = df[['Close', 'Log_returns']].values
    return df, X

def prep_intraday_data(df):
    df.drop(columns=['open', 'high', 'low', 'volume'])
    df.rename(columns={"close":"Close"})
    df['Returns'] = df.Close.pct_change()
    df['Log_returns'] = np.log(1 + df['Returns'])
    df.dropna(inplace=True)
    X = df[['Close', 'Log_returns']].values
    return df, X

def scale_data(df, X):
    

    scaler = MinMaxScaler(feature_range=(0,1)).fit(X)
    X_scaled = scaler.transform(X)
    Y = [x[0] for x in X_scaled]
    global split
    split = int(len(X_scaled)*0.8)
    X_train = X_scaled[: split]
    X_test = X_scaled[split : len(X_scaled)]
    Y_train = Y[: split]
    Y_test = Y[split : len(Y)]
    return X_test, X_train, Y_test, Y_train

def lookback_data(n, X_train, Y_train, X_test, Y_test):
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    
    for i in range(n, len(X_train)):
        Xtrain.append(X_train[i-n : i, :X_train.shape[1]])
        Ytrain.append(Y_train[i])
    for i in range(n, len(X_test)):
        Xtest.append(X_test[i-n : i, :X_test.shape[1]])
        Ytest.append(Y_test[i])
    
    Xtrain, Ytrain = (np.array(Xtrain), np.array(Ytrain))
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))
    
    Xtest, Ytest = (np.array(Xtest), np.array(Ytest))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))
    return Xtrain, Ytrain, Xtest, Ytest 
    
def LSTM_train(Xtrain, Ytrain, Xtest, Ytest):

    model = Sequential()
    model.add(LSTM(4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics='accuracy')
    model.fit(Xtrain, Ytrain, epochs=150, validation_data=(Xtest, Ytest), batch_size=16, verbose=1)
    model.summary()
    return model

def predictions(model, Xtrain, Xtest):    
    
    scaler = MinMaxScaler(feature_range=(0,1)).fit(X)
    trainPredict = np.array(model.predict(Xtrain))
    testPredict = np.array(model.predict(Xtest))
    
    trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
    testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]
    
    trainPredict = scaler.inverse_transform(trainPredict)
    trainPredict = [x[0] for x in trainPredict]
    
    testPredict = scaler.inverse_transform(testPredict)
    testPredict = [x[0] for x in testPredict]

    trainScore = mean_squared_error([x[0][0] for x in Xtrain], trainPredict, squared=False)
    print("Train Score: %.2f RMSE" % (trainScore))

    testScore = mean_squared_error([x[0][0] for x in Xtest], testPredict, squared=False)
    print('Test Score: %.2f RMSE' % (testScore))
    
    return trainPredict, testPredict

def arrange_predictions(n, df, trainPredict, testPredict):
    
    predictions = []
    for i in range(n):
        predictions.append(0)
    for x in trainPredict:
        predictions.append(x)
    for i in range(n):
        predictions.append(0)
    for x in testPredict:
        predictions.append(x)
    predictions = np.array(predictions)
    df['Predicted'] = predictions

lookback = 3
ticker = ['XLF', 'SPY', 'QQQ', 'EEM', 'GDX', 'IEMG', 'VTI', 'VOO', 'VXX', 'TVIX']
master_df = pd.DataFrame(columns=['ticker','buy_date', 'buy_price', 'stop_loss', 
                                  'sell_date', 'sell_price', 'type'])
ind_returns = []

for i in ticker:      


    #df = get_intradaydata(ticker)
    #df, X = prep_intraday_data(df)
    df, X = download_data_daily(i, dt.datetime(2018,9,1), dt.datetime(2020,10,31))
    
    X_test, X_train, Y_test, Y_train = scale_data(df, X)
    Xtrain, Ytrain, Xtest, Ytest = lookback_data(lookback, X_train, Y_train, X_test, Y_test)
    model = LSTM_train(Xtrain, Ytrain, Xtest, Ytest)
    trainPredict, testPredict = predictions(model, Xtrain, Xtest)
    arrange_predictions(lookback, df, trainPredict, testPredict)
    '''
    plt.figure(1, figsize=(16,6))
    plt.plot(df.Close, label='actual')
    plt.plot(df.Predicted[lookback:len(trainPredict)+lookback], label='training predictions')
    plt.plot(df.Predicted[len(trainPredict)+lookback*2:], label='test predictions')
    plt.legend(loc='upper left')
    plt.plot()
    plt.show()
    
    
    plt.figure(1, figsize=(16,6))
    plt.plot(df.Close[-len(testPredict):], label='actual')
    plt.plot(df.Predicted[-len(testPredict):], label='predicted')
    plt.legend(loc='upper left')
    plt.plot()
    plt.show()
    '''
    #Backtesting results
    dfbt = df.iloc[(split+lookback):]
    dfbt.reset_index(inplace=True)
    dfbt = dfbt[['Date', 'Close', 'Predicted', 'Low']]
    dfbt = backtest_daily_swing(dfbt, i)
    dfbt['Profit'] = (dfbt['sell_price'] - dfbt['buy_price'])/dfbt['buy_price']
    master_df = master_df.append(dfbt, ignore_index=True)
    print(i)
    ind_returns.append((sum(dfbt['Profit']), i))
    
    '''
    # Confidence Intervals
    
    y = df['Close']
    y_predict = df['Predicted']
    mse_l = []
    win = 5 
    for i in range(win,len(y)+1):
        mse_l.append(metrics.mean_squared_error(y[i-win:i], y_predict[i-win:i]))
    #mse_l = np.array(mse_l).reshape((len(mse_l), 1))
    num = 100
    plt.figure(1, figsize=(16,6))
    plt.plot(y[win-1:][-num:], color='r', label='actual')
    plt.plot(y_predict[win-1:][-num:], color='blue', label='predicted')
    plt.plot((y_predict[win-1:] + 1.96*np.sqrt(mse_l))[-num:], color='grey', label='upper')
    plt.plot((y_predict[win-1:] - 1.96*np.sqrt(mse_l))[-num:], color='grey', label='lower')
    plt.legend(loc = 'upper left')
    plt.plot()
    plt.show()
    
    y_pred = pd.DataFrame()
    u = y_predict[win-1:] + 1.96*np.sqrt(mse_l)
    l = y_predict[win-1:] - 1.96*np.sqrt(mse_l)
    y_pred['lower'] = l
    y_pred['upper'] = u
    y_pred = np.array(y_pred)
    y = np.array(y)
    cp = coverage_probability(y[win-1:],y_pred)
    print('coverage probability: ' + ticker + str(cp))
    '''