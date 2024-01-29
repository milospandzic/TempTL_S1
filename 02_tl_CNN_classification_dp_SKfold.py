#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchsummary import summary
from fastai.basics import *
from fastai.callbacks.tracker import EarlyStoppingCallback,SaveModelCallback

from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

from helpers_cc import *
# torch.cuda.set_device(0)

from datetime import datetime
from itertools import product


data = pd.read_csv('SAR_50points_random/50random-dataset-csv.csv')
np.unique(data['class'])

data = data[data['class'].isin([0,1,2,3,4,5,6,7,13])]
data.loc[data['class']==13,'class'] = 8

dates = data.columns[4:] 
dates = np.array([datetime.strptime(d[:-2], '%Y%m%d') for d in dates[:-1]])

best_model_rf = find_best_rf('results/hypp_opt_RF_macro.csv')
# best_model_rf_2021 = find_best_rf('results/hypp_opt_RF2021.csv')
best_model_rf_2021 = find_best_rf2('results/hypp_opt_RF_2021_with_percentage.csv')


best_model_cnn = find_best_cnn('results/hypp_opt_CNN_macro.csv')
# best_model_cnn_2021 = find_best_cnn('results/hypp_opt_CNN_blocks2021.csv')
best_model_cnn_2021 = find_best_cnn2('results/hypp_opt_CNN_2021_with_percentage.csv')

best_model_transformer = find_best_transformer('results/hypp_opt_transformer_macro.csv')

best_model_transformer_2021 = find_best_transformer2('results/hypp_opt_Transformer_2021.csv')


df_res = pd.DataFrame()
res = {}
month = [('September',105)]
points_of_interest = np.array([4, 9, 14, 19, 24, 29])
conf_list = []
nconf_list = []

average_type = 'macro'

for iteration in range(5):

    print("Iteracija broj:")
    print(iteration)
        
    idx_col = np.where(dates>datetime(2017, 3, 1))[0][:month[0][1]] + 4 
    data_new = data.iloc[:,[0,1,2]+list(idx_col)] 

    data_new.columns = data_new.columns.str.replace('2017','')  
    data_new['class'] = data['class']

    df_res, conf_list, nconf_list = initial2(data_new, best_model_cnn, best_model_cnn_2021, best_model_rf, best_model_rf_2021, best_model_transformer, best_model_transformer_2021, res, df_res, average_type, iteration, month, points_of_interest, conf_list, nconf_list)


df_res







