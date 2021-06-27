# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:59:49 2021

@author: Anshumaan
"""

#Alphavantage API key: B45I3T6LI9HCUDCT
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

def get_intradaydata(ticker):
    
    apikey = 'B45I3T6LI9HCUDCT'
    app = TimeSeries(apikey, output_format='csv')
    slices = ['year1month1', 'year1month2', 'year1month3', 'year1month4']
    df_ticker = pd.Dataframe()
    
    for timeslice in slices:
        df, meta = app.get_intraday_extended(symbol=ticker, interval='60min', slice=timeslice)
        df = pd.DataFrame(df)
        df.columns = df.iloc[0]
        df = df.set_index('time')
        df_ticker.append(df)
    
    return df_ticker