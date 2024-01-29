#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from helpers_cc import *

from datetime import datetime
from itertools import product
import time

from joblib import dump


def train_model(X_train, Y_train, X_val, Y_val, params, year):
    
    res = {}
    
    model = RandomForestClassifier(n_estimators = params[0], max_depth = params[1], min_samples_leaf = params[2], warm_start=True)
    
    start_time = time.time()
    model.fit(X_train, Y_train)
    time_comp = time.time()-start_time
    
    dump(model, f'models/RF-year-{year}-n_estimators-{params[0]}-max_depths-{params[1]}-min_samples_leaf-{params[2]}-percentage-{params[3]}.joblib')                   
    
    predictions = model.predict(X_val)
    
    res['accuracy'] = accuracy_score(Y_val, predictions)
    res['F1 score'] = f1_score(Y_val, predictions,average = 'macro')
    res['precision'] = precision_score(Y_val, predictions,average = 'macro')
    res['recall'] = recall_score(Y_val, predictions,average = 'macro')
    
    res['Time complexity'] = time_comp

    return res

data = pd.read_csv('SAR_50points_random/50random-dataset-csv.csv')
np.unique(data['class'])


data = data[data['class'].isin([0,1,2,3,4,5,6,7,13])]
data.loc[data['class']==13,'class'] = 8

dates = data.columns[4:] 
dates = np.array([datetime.strptime(d[:-2], '%Y%m%d') for d in dates[:-1]])
idx = np.where(dates>datetime(2017, 3, 1))[0][:105] + 4 
data_new = data.iloc[:,[0,1,2]+list(idx)] 

data_new.columns = data_new.columns.str.replace('2017','') 
data_new['class'] = data['class']
data_new.head()


n_estimators = [100, 500, 1000]
max_depths = [1, 5, 10, 50]
min_samples_leaf = [1, 5, 10, 50, 100]
percentage = [5, 10, 15, 20, 25, 30]

params = list(product(n_estimators, max_depths, min_samples_leaf, percentage))


df_res = pd.DataFrame()
month = [('September',105)]

# for y in np.unique(data_new.year)[:-1]: # all seasons but last
for y in np.unique(data_new.year)[4:]: # only last season
    for m in month:
        for p in params:
            
            idx = np.where(dates>datetime(2017, 3, 1))[0][:m[1]] + 4 
            data_new = data.iloc[:,[0,1,2]+list(idx)] 

            data_new.columns = data_new.columns.str.replace('2017','') 
            data_new['class'] = data['class']

            valid_data = data_new.loc[data_new.year==2021]
            valid_per_field_all = valid_data.groupby('ID_p').mean()

            X = valid_per_field_all.iloc[:,:-1]
            Y = valid_per_field_all.iloc[:,-1]
            X_prim, X_test, Y_prim, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.6, random_state=42)
            X_train, X_val, Y_train, Y_val = train_test_split(X_prim, Y_prim, stratify=Y_prim, test_size=0.25, random_state=42)

            """
            Increments_start
            """
            if (p[3]!=30):
                X_trainHELP, X_valHELP, Y_trainHELP, Y_valHELP = train_test_split(X_train, Y_train, stratify=Y_train, test_size=(1-p[3]/30), random_state=42)
                X_train, Y_train = X_trainHELP, Y_trainHELP
            """
            Increments_end
            """
            train_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_train.index])
            # test_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_test.index])
            validation_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_val.index])

            valid_data_learn = valid_data.loc[train_sample_idx, :] # X_train
            valid_data_validate = valid_data.loc[validation_sample_idx, :] #

            X_train = valid_data_learn.iloc[:,3:-1]
            Y_train = valid_data_learn.iloc[:,-1]

            X_val = valid_data_validate.iloc[:,3:-1]
            Y_val = valid_data_validate.iloc[:,-1]

            res = train_model(X_train, Y_train, X_val, Y_val, p, y)
            
            res['Validation year'] = int(y)
            res['Month'] = m[0]
            res['Params'] = str(p)

            df_res = df_res.append(pd.DataFrame(res,index = [0]))
            df_res.to_csv('results/hypp_opt_RF_2021_with_percentage.csv')

df_res


