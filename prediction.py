#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:13:55 2020

@author: ryancrisanti
"""

from operator import attrgetter
import pandas as pd
import numpy as np
import warnings
from .delta import Delta
from .utilities import is_unique, format_name, month_range, save, load

class Prediction:
    # TODO: Add in a total margin of error
    def __init__(self, name, start_date, start_money):
        self.name = format_name(name)
        self.start_date = start_date
        self.start_money = start_money
        self._deltas = []
    
    def add_delta(self, delta):
        if not isinstance(delta, Delta):
            raise TypeError
            
        if is_unique(delta, self.deltas):
            self._deltas.append(delta)
        else:
            raise ValueError(f'`delta` ({delta.name}) must have unique name')
    
    @property
    def deltas(self):
        return self._deltas
    
    @property
    def named_deltas(self):
        return {d.name: d for d in self.deltas}
    
    def add_deltas(self, deltas):
        for delta in deltas:
            self.add_delta(delta)
    
    def remove_delta(self, name):
        get_name = attrgetter('name')
        index = [i for i,d in enumerate(self.deltas) if name==get_name(d)]
        if len(index)==0:
            raise ValueError(f'No Delta with name "{name}" found.')
        elif len(index)==1:
            index = index[0]
            _ = self._deltas.pop(index)
            print(f'Removed Delta "{name}" at index {index}.')
        else:
            raise ValueError(
                'Should not ever get this, but found {len(index)} Deltas '\
                f'with name "{name}", should not be more than 1.')
    
    def remove_deltas(self, names):
        for name in names:
            self.remove_delta(name)
    
    def change_delta_value(self, name, newvalue):
        self.named_deltas[name].value = newvalue
    
    def project(self, end_date=pd.Timestamp(pd.Timestamp.today().date()), 
                 time_granularity='W', dates=None):
        '''
        To get month start, use "MS"
        To get month end, use "M"
        To get month from start_date, use "Mcustom"
        '''
        index = self._decipher_sim_params(end_date, time_granularity, dates)
        df = pd.DataFrame(index=index, columns=self.named_deltas.keys())
        df.index.name = 'TimeInterval'
        # Make copies to use later in calculations of uncertainty
        df_posunc = df.copy()
        df_negunc = df.copy()
        
        # Fill dataframe
        df = df.apply(self._count_occurances, axis=1, value_attr='value')
        
        # Sum up
        df['IntervalTotalDelta'] = df.sum(axis=1)
        self.totals = self._get_totals(self.start_money, df)
        df['IntervalEndTotal'] =  self.totals[1:]
        self.df_worksheet = df
        
        self.df_tot = self._build_totals_df(self.totals, 
                                            self.df_worksheet.index)
        
        # Now, fill the uncertainty DFs
        df_posunc = df_posunc.apply(self._count_occurances, axis=1, 
                                    value_attr='uncertainty_pos')
        df_negunc = df_negunc.apply(self._count_occurances, axis=1, 
                                    value_attr='uncertainty_neg')
        posunc = self._get_uncertainty_series(df_posunc, 
                                              name='PositiveUncertainty')
        negunc = self._get_uncertainty_series(df_negunc,
                                              name='NegativeUncertainty')
        self.df_unc = pd.DataFrame([posunc, negunc]).T
        # Now, make it in terms of the actual upper & lower bounds of possible
        # $ instead of delta
        df_bounds = self.df_tot.copy().join(self.df_unc).fillna(0)
        df_bounds['LowBound'] = df_bounds[self._COLUMN_NAME]\
            -df_bounds['NegativeUncertainty']
        df_bounds['HighBound'] = df_bounds[self._COLUMN_NAME]\
            +df_bounds['PositiveUncertainty']
        self.df_bounds = df_bounds.drop(columns=['NegativeUncertainty', 
                                                 'PositiveUncertainty'])
    
    def _decipher_sim_params(self, end_date, time_granularity, dates):
        if dates is None:
            if not isinstance(end_date, pd.Timestamp):
                raise TypeError
            if not isinstance(time_granularity, str):
                raise TypeError
            
            index = self._build_tindex(end=end_date, freq=time_granularity)
            self.end_date = end_date
            self.time_granularity = time_granularity
        else:
            if not all([isinstance(e, pd.Timestamp) for e in dates]):
                raise TypeError('All elements of `dates` must be pd.Timestamps')
            if not all([p is None for p in [end_date, time_granularity]]):
                warnings.warn(
                    'Since explicit `dates` was passed, ignoring the '\
                    '`end_date` and `time_granularity` parameters.')
            trange = sorted(list(dates))
            index = self._build_tindex(trange=trange)
            self.time_granularity = 'custom'
        return index
    
    def _build_tindex(self, end=None, freq=None, trange=None):
        if trange is None and all([p is not None for p in [end, freq]]):
            if freq == 'Mcustom':
                trange = month_range(self.start_date, end)
            else:
                trange = pd.date_range(start=self.start_date, end=end, freq=freq)
        elif trange is not None and all([p is None for p in [end, freq]]):
            trange = sorted(list(np.unique(trange)))
            end = trange[-1]
            self.end_date = end
        else:
            raise ValueError('Must pass either `trange` or all of [`end`, '\
                             '`freq`], but not all 3.')
        tintervals = [pd.Interval(trange[i], trange[i+1], closed='right') 
                      for i in range(len(trange)-1)]
        
        if not any([self.start_date in intv for intv in tintervals]):
            tintervals = [
                pd.Interval(self.start_date, tintervals[0].left, closed='both')
                ]+tintervals
        
        if not any([end in intv for intv in tintervals]):
            tintervals.append(
                pd.Interval(tintervals[-1].right, end, closed='right')
                )

        return tintervals

    def _count_occurances(self, row, value_attr='value'):
        for col in row.index:
            delta = self.named_deltas[col]
            count = 0
            for date in delta.dates:
                if date in row.name:
                    count += 1
            row[col] = count * getattr(delta, value_attr) #delta.value
        return row
    
    def _get_totals(self, start_money, df):
        tot = [start_money]
        for d in df.IntervalTotalDelta:
            tot.append(tot[-1]+d)
        return np.array(tot)
    
    def _build_totals_df(self, totals, index):
        idx = np.array([self.start_date]+[idx.right for idx in index])
        self._COLUMN_NAME = f'Pred "{self.name}" Balance'
        df = pd.DataFrame(totals, index=idx, columns=[self._COLUMN_NAME])
        df = df[~df.index.duplicated(keep='last')]
        return df
    
    def _get_uncertainty_series(self, df_unc, name=''):
        ser = df_unc.sum(axis=1).cumsum(axis=0)
        ser.index = [idx.right for idx in ser.index]
        ser.name = name
        return ser
    
    def save(self, filepath, **kwargs):
        save(self, filepath, **kwargs)

def load_prediction(filepath, **kwargs):
    return load(filepath, **kwargs)
    