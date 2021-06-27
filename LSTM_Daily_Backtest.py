# -*- coding: utf-8 -*-
"""
Created on Thu May 27 19:14:03 2021

@author: Anshumaan Gandhi
"""
import datetime as dt
import pandas as pd
import numpy as np

def backtest_daily_swing(df, symbol):
    
    dfbt = pd.DataFrame(columns=['ticker','entry_date', 'buy_price', 'stop_loss', 'exit_date', 'sell_price', 'type'])
    td = {'ticker': 0, 'entry_date': 0, 'buy_price': 0, 'stop_loss': 0, 'exit_date': 0, 'sell_price': 0, 'type': 0}
    flag = 0
    
    f = open('tradelog_dlyswing.txt', 'w+')
    
    #df = df.reset_index(inplace=True)
    X = np.array(df)
    f.write('beginning trading backtest on: ')
    f.write(str(X[0][0]))
    f.write('\n')
    for i in range(len(X)-1):
        
        
        if flag == 1:
            
            if X[i][1] < X[i][2]:
                #Sell at X[i][2]
                flag = 0
                f.write('stop loss triggered on long')
                f.write(str(X[i][0]))
                f.write('\n')
                td['exit_date'] = X[i][0]
                td['sell_price'] = X[i][2]*0.9965
                td['type'] = 'Long'
                dfbt = dfbt.append(td, ignore_index=True)
                #dfbt = dfbt.append({'sell_date': X[i][0],
                #                    'sell_price': X[i][2]*0.9965}, ignore_index=True)
            
            else:
                #Hold Positions
                f.write('Holding Positions on: ')
                f.write(str(X[i][0]))
                f.write('\n')
        
        elif flag == 0:
            
            if X[i][1] > X[i][2] and X[i+1][2] > X[i][2]:
                #Initiate long position
                flag = 1
                f.write('Initiating long on: ')
                f.write(str(X[i][0]))
                f.write('\n')
                td['ticker'] = symbol
                td['entry_date'] = X[i][0]
                td['buy_price'] = X[i][1]
                td['stop_loss'] = X[i][2]*0.9965
                #dfbt = dfbt.append({'buy_date': X[i][0], 
                #                   'buy_price': X[i][1],
                #                   'stop_loss': X[i][2]*0.9965}, ignore_index=True)
            
            if X[i][1] < X[i][2] and X[i+1][2] < X[i][2]:
                # Initiating a short
                flag = -1
                f.write('Initiating Short on: ')
                f.write(str(X[i][0]))
                f.write('\n')
                td['ticker'] = symbol
                td['entry_date'] = X[i][0]
                td['sell_price'] = X[i][1]
                td['stop_loss'] = X[i][2]*1.0035  
        
        else:
            #how to handle short positions
            if X[i][1] > X[i][2]:
                #Sell at X[i][2]
                flag = 0
                f.write('stop loss triggered on short')
                f.write(str(X[i][0]))
                f.write('\n')
                td['exit_date'] = X[i][0]
                td['buy_price'] = X[i][2]*1.0035
                td['type'] = 'Short'
                dfbt = dfbt.append(td, ignore_index=True)
                #dfbt = dfbt.append({'sell_date': X[i][0],
                #                    'sell_price': X[i][2]*0.9965}, ignore_index=True)
            
            else:
                #Hold Positions
                f.write('Holding Positions on: ')
                f.write(str(X[i][0]))
                f.write('\n')
              
            
            
    f.close()
    
    return dfbt
                