#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:36:44 2020

@author: ryancrisanti
"""

import pandas as pd
import pickle
import os

def format_money_value(value):
    ret = f'${abs(value):,.2f}'
    if value < 0:
        ret = f'({ret})'
    return ret

def format_name(name):
    return name.strip().title().replace(' ', '')

def is_unique(obj, objs, idfy='name'):
    return not any([getattr(obj, idfy)==getattr(o, idfy) for o in objs])

def month_range(start, end):
    rng = pd.date_range(start-pd.offsets.MonthBegin(), end=end, freq='MS')
    ret = (rng + pd.offsets.Day(start.day-1)).to_series()
    ret.loc[ret.dt.month > rng.month] -= pd.offsets.MonthEnd(1)
    # return pd.DatetimeIndex(ret)
    # made this adjustment to account for inconsistencies
    return pd.DatetimeIndex(ret.loc[start:end]) 

def save(obj, filepath, **kwargs):
    filepath = _check_filepath_ext(filepath, expected='.pickle')
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, **kwargs)

def load(filepath, **kwargs):
    filepath = _check_filepath_ext(filepath, expected='.pickle')
    with open(filepath, 'rb') as f:
        obj = pickle.load(f, **kwargs)
    return obj

def _check_filepath_ext(filepath, expected='.pickle'):
    filepath = os.path.abspath(filepath)
    fname, ext = os.path.splitext(filepath)
    if ext == expected:
        pass
    elif ext == '':
        filepath += expected
    else:
        raise ValueError(f'File cannot have extension other than "{expected}'\
                         f'", but got extension "{ext}".')
    return filepath

def check_delta_inputs(value, dates, name):
    msg = 'Expected `{exp}` to be of type "{exptype}", but got "{got}" '\
        'of type "{gottype}" instead'
    if not isinstance(value, (float, int)):
        raise TypeError(msg.format(exp='value', exptype='float/int', 
                                   got=value, gottype=type(value)))
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError(msg.format(
            exp='dates', exptype='pd.DatetimeIndex', 
            got=dates, gottype=type(dates)))
    if not isinstance(name, str):
        raise TypeError(msg.format(exp='name', exptype='str', 
                                   got=name, gottype=type(name)))