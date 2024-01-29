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

from datetime import datetime
from itertools import product


data = pd.read_csv('SAR_50points_random/50random-dataset-csv.csv')
np.unique(data['class'])

data = data[data['class'].isin([0,1,2,3,4,5,6,7,13])]
data.loc[data['class']==13,'class'] = 8

dates = data.columns[4:] 
dates = np.array([datetime.strptime(d[:-2], '%Y%m%d') for d in dates[:-1]])

best_model_rf_2021 = find_best_rf2('results/hypp_opt_RF_2021_with_percentage.csv')

best_model_cnn = find_best_cnn('results/hypp_opt_CNN_macro.csv')

def initial3(data_new, best_model_cnn, best_model_rf_2021, res, df_res, average_type, iteration, month, points_of_interest):

    valid_data = data_new.loc[data_new.year==2021]
    valid_per_field_all = valid_data.groupby('ID_p').mean()
    
    X = valid_per_field_all.iloc[:,:-1]
    Y = valid_per_field_all.iloc[:,-1]
    X_prim, X_test, Y_prim, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.6, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_prim, Y_prim, stratify=Y_prim, test_size=0.25, random_state=42)
    train_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_train.index])
    test_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_test.index])

    valid_data_learn = valid_data.loc[train_sample_idx, :] # AKA valid_data_learn = valid_data.loc[test_fold_sample_idx, :]
    valid_data_validate = valid_data.loc[test_sample_idx, :] # AKA valid_data_validate = valid_data.drop(index=test_fold_sample_idx)

    print(train_sample_idx)

    valid_data_learn_fields = valid_data_learn.groupby('ID_p').mean()
    valid_data_learn_fields_classes = np.unique(valid_data_learn_fields['class'])

    for idx, step in enumerate(np.round(np.arange(0.0333333333333333, 1+0.00000000000001, 0.0333333333333333333),3)[points_of_interest]): # 1%,2%,3%,...,29%,30%

        valid_data_learn_small = pd.DataFrame([])

        for tc in valid_data_learn_fields_classes.astype(int):
            random.seed(iteration)
            test_fold_per_field_idx = list(valid_data_learn_fields.loc[valid_data_learn_fields['class']==tc,:].index)
            if step == 1:
                test_per_field_step_idx = test_fold_per_field_idx
            else:
                test_per_field_step_idx = random.sample(test_fold_per_field_idx, int(np.ceil(len(test_fold_per_field_idx)*step)))

            test_sample_step_idx = np.hstack([valid_data_learn.loc[valid_data_learn['ID_p']==t,:].index for t in test_per_field_step_idx])
            valid_data_learn_small = pd.concat([valid_data_learn_small, valid_data_learn.loc[test_sample_step_idx, :]])

        X_test = torch.Tensor(valid_data_validate.iloc[:,3:-1].to_numpy())
        Y_test = torch.Tensor(valid_data_validate.iloc[:,-1].to_numpy())
        X_test = X_test.reshape(-1, 1, month[0][1]) 

        X_train_percent = torch.Tensor(valid_data_learn_small.iloc[:,3:-1].to_numpy())
        Y_train_percent = torch.Tensor(valid_data_learn_small.iloc[:,-1].to_numpy())
        X_train_percent = X_train_percent.reshape(-1, 1, month[0][1]) 
                    
        data_bunch_tl = create_databunch(X_train_percent, X_train_percent, Y_train_percent.long(), Y_train_percent.long(), bs=512)
        
        res['Iteration'] = iteration
        res['Training ratio'] = int(np.round(step*30))

        res = cnn_learning3(None, data_bunch_tl, X_test, Y_test, best_model_cnn, res, average_type, iteration)               
        res = rf_learning3(X_train_percent, Y_train_percent, X_test, Y_test, best_model_rf_2021[idx], res, average_type, iteration)

        df_res = df_res.append(pd.DataFrame(res,index = [0]))
        df_res.to_csv('Initial2_results_5-CNN-TL_30-RF-2021.csv')

    return df_res

def cnn_learning3(data_bunch_create, data_bunch_tl, X_test, Y_test, best_model_cnn, res, average_type, iteration):

    if data_bunch_create is not None:
        set_seed(42+iteration)
        learn = create_learner(data_bunch_create, nfilters=eval(best_model_cnn.Params)[0], kernel_size = eval(best_model_cnn.Params)[1], drop_prob = eval(best_model_cnn.Params)[2])
    else:
        set_seed(42+iteration)
        learn = create_learner(data_bunch_tl, nfilters=eval(best_model_cnn.Params)[0], kernel_size = eval(best_model_cnn.Params)[1], drop_prob = eval(best_model_cnn.Params)[2])
        
    learn.load(f'model-year-{best_model_cnn["Validation year"]}-nfilters-{eval(best_model_cnn.Params)[0]}-kernel_size-{eval(best_model_cnn.Params)[1]}-drop_prob-{eval(best_model_cnn.Params)[2]}')
    learn.model.cuda()
    
    learn.data = data_bunch_tl
    learn.freeze_to(-1)
    learn.model.train()

    start_time = time.time()
    learn.fit_one_cycle(50,max_lr=1e-3,callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss', name= 'model'), ])
    time_comp = time.time()-start_time

    learn.model.eval()
    predictions_cnn = np.argmax(learn.model(torch.Tensor(X_test).cuda()),axis=1)

    res['F1 score CNN'] = f1_score(Y_test, predictions_cnn, average = average_type)
    res['Precision score CNN'] = precision_score(Y_test, predictions_cnn, average = average_type)
    res['Recall score CNN'] = recall_score(Y_test, predictions_cnn, average = average_type)
    res['Accuracy CNN'] = accuracy_score(Y_test, predictions_cnn)

    res['F1 score per class CNN'] = str(f1_score(Y_test, predictions_cnn, average = None))
    res['Precision score per class CNN'] = str(precision_score(Y_test, predictions_cnn, average = None))
    res['Recall score per class CNN'] = str(recall_score(Y_test, predictions_cnn, average = None))

    res['Time complexity CNN'] = time_comp

    return res

def rf_learning3(X_train_percent, Y_train_percent, X_test, Y_test, best_model_rf_2021, res, average_type, iteration):

    rf_fs_2021 = RandomForestClassifier(n_estimators=eval(best_model_rf_2021.Params)[0], max_depth=eval(best_model_rf_2021.Params)[1], min_samples_leaf=eval(best_model_rf_2021.Params)[2], random_state=42+iteration)
    start_time = time.time()
    rf_fs_2021.fit(np.squeeze(X_train_percent), Y_train_percent)
    time_comp = time.time()-start_time
    predictions_rf = rf_fs_2021.predict(np.squeeze(X_test))

    res['fsF1 score RF 2021'] = f1_score(Y_test, predictions_rf, average = average_type)
    res['fsPrecision score RF 2021'] = precision_score(Y_test, predictions_rf, average = average_type)
    res['fsRecall score RF 2021'] = recall_score(Y_test, predictions_rf, average = average_type)
    res['fsAccuracy RF 2021'] = accuracy_score(Y_test, predictions_rf)

    res['fsF1 score per class RF 2021'] = str(f1_score(Y_test, predictions_rf, average = None))
    res['fsPrecision score per class RF 2021'] = str(precision_score(Y_test, predictions_rf, average = None))
    res['fsRecall score per class RF 2021'] = str(recall_score(Y_test, predictions_rf, average = None))

    res['fsTime complexity RF 2021'] = time_comp

    return res

df_res = pd.DataFrame()
res = {}
month = [('September',105)]
points_of_interest = np.array([4, 29])
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
    
    df_res  = initial3(data_new, best_model_cnn, best_model_rf_2021, res, df_res, average_type, iteration, month, points_of_interest)
    
df_res







