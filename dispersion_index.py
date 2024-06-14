#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:21:51 2023

@author: pmaresnasarre
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def DI_plot(sample, dates, th, significance):
# sample: original sample, pandas dataframe
# dates: datetime of the date of the observation
# dist: distance between independent events [equal to the sampling time]
#[REF] Cunnane (1979): A note on the Poisson assumption in partial duration series models.

    st     = np.quantile(sample, .995)/45
    th_min = np.quantile(sample, .75)
    # th_min = 0
    th_max = sample.max()
    th  = np.arange(th_min, th_max, step = st, dtype='float32')
    
    n_years = len(np.unique(dates.dt.year))
    dist = np.linspace(12, 96, 50)
    #Initialization of some main arrays
    di = []  #shape parameter intialization 
    for u in dist:        
    
        # Find peaks 
        pks, _ = find_peaks(sample, height = th, distance=u) # pks indices
        ex   = sample.iloc[pks] - u
        d_ex = dates.iloc[pks]
        
        file2 = pd.concat([d_ex.rename('date'), ex.rename('var')], axis = 1)
        file2.set_index('date', inplace=True)

        
        nExceedances = file2.groupby(lambda x: file2.index.year).count()    
        
        e_mean = nExceedances.mean()
        e_var = nExceedances.var()
        
        di.append(e_var/e_mean)
        
    
    CI_up = chi2.isf(significance/2, n_years-1)/(n_years-1)
    CI_down = chi2.isf(1-significance/2, n_years-1)/(n_years-1)

    
    #Plotting shape parameter against u vales   
    fig, ax = plt.subplots(1,1,figsize=(10,4))   
    ax.plot(th, di, color = "k")
    plt.fill_between(th, [CI_down]*len(th), [CI_up]*len(th), alpha = 0.4)
    ax.set_xlabel('Dl')
    ax.set_ylabel(r'$DI$')
    ax.set_title('Dispersion Index plot')
    ax.set_xlim([th_min, th_max])

    ax.grid()
    
    plt.show()
    
    return