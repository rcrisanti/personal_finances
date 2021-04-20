#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:20:06 2020

@author: ryancrisanti
"""

from .utilities import format_name, format_money_value, save, load
import pandas as pd
import os

class Account:
    COLUMN_NAME = 'Value'
    INDEX_NAME = 'Date'
    
    def __init__(self, name):
        self.name = format_name(name)
        self.value = pd.DataFrame(columns=[self.COLUMN_NAME])
        self.value.index.name = self.INDEX_NAME
    
    def __repr__(self):
        if self.value.shape[0] == 0:
            return '<Account: no balance recorded>'
        else:
            return f'<Account: {format_money_value(self.value.iloc[-1].item())}'\
                f' on {self.value.index[-1].date()}>'
    
    def __str__(self):
        if self.value.shape[0] == 0:
            return '<Account: no balance recorded>'
        else:
            return f'<Account: {format_money_value(self.value.iloc[-1].item())}'\
                f' on {self.value.index[-1].date()}>'
    
    def update(self, updates, datecol='Date', balancecol='Balance', **kwargs):
        if isinstance(updates, str):
            # Check if the string is a path location
            if os.path.exists(os.path.abspath(updates)):
                updates = self._update_read(filepath=updates, datecol=datecol, 
                                            balancecol=balancecol, **kwargs)
            else:
                raise TypeError('Can only interpret strings as file paths, and '\
                          'this one doesn\'t exist')

        # Do a bunch of type checking
        if isinstance(updates, pd.DataFrame):
            updates = updates.copy()
            if isinstance(updates.index, pd.DatetimeIndex):
                if updates.shape[1] == 1:
                    updates = self._update_df(updates)
                else:
                    raise ValueError('More than 1 column')
            else:
                raise TypeError('Not a datetime index')
        else:
            raise TypeError('Not a DataFrame')
        
        # Update the values DF
        self.value = pd.concat([self.value, updates])
        self.value = self.value[~self.value.index.duplicated(keep='last')]
        self.value.sort_index(inplace=True)
    
    def _update_df(self, df):
        df.columns = [self.COLUMN_NAME]
        df.index.name = self.INDEX_NAME
        df.index = df.index.map(lambda x: pd.Timestamp(x.date()))
        return df
    
    def _update_read(self, filepath, datecol, balancecol, **kwargs):
        _, ext = os.path.splitext(filepath)
        if ext == '.csv':
            df = pd.read_csv(filepath, **kwargs)
        elif ext == '.xlsx':
            df = pd.read_excel(filepath, **kwargs)
        else:
            raise TypeError('Can only read csv & xlsx files')
        datefmt  = lambda s: pd.Timestamp(s)
        moneyfmt = lambda s: float(s.replace('$', '').replace(',', ''))
        df = pd.DataFrame(data=df[balancecol].apply(moneyfmt).values, 
                          index=pd.DatetimeIndex(df[datecol].apply(datefmt)))
        return df
    
    def remove(self, dates):
        self.value = self.value.drop(labels=dates)
    
    def save(self, filepath, **kwargs):
        save(self, filepath, **kwargs)

def load_account(filepath, **kwargs):
    return load(filepath, **kwargs)
    
    
    