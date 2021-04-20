#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:06:04 2020

@author: ryancrisanti
"""

from .prediction import Prediction
from .account import Account
from .utilities import format_name, save, load
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Comparison:
    def __init__(self, predictions=[], accounts=[], name=''):
        self.name = format_name(name)
        self._predictions = list(predictions)
        self._accounts = list(accounts)
    
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def named_predictions(self):
        return {p.name: p for p in self.predictions}
    
    @property
    def accounts(self):
        return self._accounts
    
    @property
    def named_accounts(self):
        return {a.name: a for a in self.accounts}
    
    @property
    def acnt_values(self):
        if len(self.accounts) == 0:
            return pd.DataFrame()
        else:
            return pd.concat([acc.value for acc in self.accounts], axis=1)
    
    @property
    def tot_acnt_value(self):
        '''Resolves unmatched dates with forward filling'''
        vals = self.acnt_values.fillna(method='ffill').dropna().sum(axis=1)
        vals.rename('TotalAccountsValue', inplace=True)
        return vals
    
    def add_prediction(self, pred):
        if not isinstance(pred, Prediction):
            raise TypeError
        self._predictions.append(pred)
        
    def add_predictions(self, preds):
        for pred in preds:
            self.add_prediction(pred)
    
    def add_account(self, acct):
        if not isinstance(acct, Account):
            raise TypeError
        self._accounts.append(acct)
    
    def add_accounts(self, accts):
        for acct in accts:
            self.add_account(acct)
    
    def pred_diffs(self, date_method='linear', linear_freq='D', 
                   linear_start='account', linear_end='account'):
        '''
        

        Parameters
        ----------
        date_method : str, optional
            Can be either "linear" or "account".
                * linear (by day, week, month, etc.)
                * account (use dates in tot_acnt_value index)
            The default is 'linear'.
        linear_freq : str, optional
            If `date_method` is "linear", this determines the frequency. 
            Otherwise, this is ignored. The default is 'D'.
        linear_start : pd.Timestamp or 'account', optional
            If `date_method` is "linear", this determines the start date. 
            Keyword 'account' means to use the first date in tot_acnt_value 
            index. The default is 'account'.
        linear_end : pd.Timestamp or 'account', optional
            If `date_method` is "linear", this determines the end date. 
            Keyword 'account' means to use the last date in tot_acnt_value 
            index. The default is 'account'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        diffs : TYPE
            DESCRIPTION.

        '''
        if date_method == 'linear':
            if linear_start == 'account':
                linear_start = self.tot_acnt_value.index.min()
            if linear_end == 'account':
                linear_end = self.tot_acnt_value.index.max()
            index = pd.date_range(linear_start, linear_end, freq=linear_freq)
        elif date_method == 'account':
            index = self.tot_acnt_value.index
        else:
            raise ValueError
                
        # Simulate all predictions
        for pred in self.predictions:
            pred.project(dates=index, end_date=None, time_granularity=None)
        
        # Make df with all preds & total acct
        df = pd.DataFrame(self.tot_acnt_value).join(
            [pred.df_tot for pred in self.predictions], 
            how='outer'
            ).fillna(method='ffill')
        df.columns = [self.tot_acnt_value.name]+[
            pred.name for pred in self.predictions]
        df = df.loc[index,:]
        
        # Get differences
        meas = df[self.tot_acnt_value.name]
        diffs = df.drop(
            columns=[self.tot_acnt_value.name]
            ).subtract(meas, axis=0) * -1  # negate to get the right sign
        diffs = diffs.loc[self.tot_acnt_value.index, :]
        self.df_preddiff = diffs
        return diffs
    
    def plot_values(self, plot_type='step', ax=None, in_subplot=False, 
                    plot_uncertainty=True, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
            
        fill_kwargs = {'alpha': .5 * kwargs.get('alpha', 1)}
        if plot_type == 'classic':
            ax.plot(self.tot_acnt_value, label='Total account value', **kwargs)
        elif plot_type == 'step':
            where = kwargs.pop('where', 'post')
            ax.step(self.tot_acnt_value.index, self.tot_acnt_value.values, 
                    label='Total account value', where=where, **kwargs)
            fill_kwargs['step'] = where
        else:
            raise ValueError('`plot_type` must be either "step" or "classic",'\
                             f' but got "{plot_type}".')
        
        for name,pred in self.named_predictions.items():
            if plot_type == 'classic':
                p = ax.plot(pred.df_tot, label=f'Prediction: {name}', **kwargs)
            elif plot_type == 'step':
                p = ax.step(pred.df_tot.index, pred.df_tot.values, 
                            label=f'Prediction: {name}', where=where, **kwargs)
            else:
                raise ValueError('`plot_type` must be either "step" or '\
                                 f'"classic", but got "{plot_type}".')
            # Fill area for uncertainty
            if plot_uncertainty:
                ax.fill_between(
                    x=pred.df_bounds.index, 
                    y1=pred.df_bounds['LowBound'].values,
                    y2=pred.df_bounds['HighBound'].values,
                    color=p[0].get_color(),
                    **fill_kwargs
                    )
                    
        ax.legend(fontsize=7)
        formatter = ticker.FuncFormatter(lambda x, p: f'${x:,.2f}')
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(lw=.5)
        if not in_subplot:
            ax.set_xlabel('Date')
        ax.set_ylabel('Value [$]')
        ax.set_title('Prediction & Actual Account Value')
        return ax
    
    def plot_difference(self, plot_type='step', ax=None, in_subplot=False, 
                        **kwargs):
        if ax is None:
            _, ax = plt.subplots()
            
        if in_subplot:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors.append(colors.pop(0))
            ax.set_prop_cycle(color=colors)
        
        if plot_type == 'step':
            where = kwargs.pop('where', 'post')
            
        for col in self.df_preddiff.columns:
            if plot_type == 'classic':
                ax.plot(self.df_preddiff[col], label=col, **kwargs)
            elif plot_type == 'step':
                ax.step(
                    self.df_preddiff[col].index, self.df_preddiff[col].values, 
                    where=where, label=col, **kwargs)
            else:
                raise ValueError('`plot_type` must be either "step" or '\
                                 f'"classic", but got "{plot_type}".')
                
        ax.legend(fontsize=7)
        formatter = ticker.FuncFormatter(lambda x, p: f'${x:,.2f}')
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(lw=.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Difference [$]')
        ax.set_title('Prediction Error (meas - pred)')
        return ax
    
    def plot(self, plot_type='step', figsize=(7,5), sharex=True, sharey=False, 
             plot_uncertainty=True, **kwargs):
        '''
        

        Parameters
        ----------
        plot_type : str, optional
            Can be either "step" or "classic". The default is 'step'.
        figsize : TYPE, optional
            DESCRIPTION. The default is (7,5).
        sharex : TYPE, optional
            DESCRIPTION. The default is True.
        sharey : TYPE, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.

        '''
        fig, ax = plt.subplots(2, figsize=figsize, sharex=sharex, sharey=sharey)
        self.plot_values(ax=ax[0], plot_type=plot_type, in_subplot=True, 
                         plot_uncertainty=plot_uncertainty, **kwargs)
        self.plot_difference(ax=ax[1], plot_type=plot_type, in_subplot=True, 
                             **kwargs)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig
    
    def save(self, filepath, **kwargs):
        save(self, filepath, **kwargs)

def load_comparison(filepath, **kwargs):
    return load(filepath, **kwargs)
        
    
    