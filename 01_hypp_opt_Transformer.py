#!/usr/bin/env python
# coding: utf-8

# In[9]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchsummary import summary
from fastai.basics import *
from fastai.callbacks.tracker import EarlyStoppingCallback, SaveModelCallback

from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from helpers_cc import *
torch.cuda.set_device(1)

from datetime import datetime
from itertools import product
import time

from breizhcrops import TransformerModel


# In[11]:


def set_seed(x=42): 
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)


# In[17]:
def create_transformer_learner(data, d_model, n_head, n_layers, d_inner, drop_prob, model = None):
    
    model = TransformerModel(input_dim=105, num_classes=9, d_model=d_model, n_head=n_head, n_layers=n_layers, d_inner=d_inner, activation="relu", dropout=drop_prob)
    
    average_type = 'macro'
    learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy,FBeta(average=average_type, beta=1),Precision(average=average_type), Recall(average=average_type)],callback_fns=[ShowGraph,partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=15)])
          
    print('Number of parameters: ',sum(p.numel() for p in model.parameters()))
          
    return learn



def train_model(data, params, year):
    
    set_seed()
    learn = create_transformer_learner(data, d_model=params[0], n_head=params[1], n_layers=params[2], d_inner=params[3], drop_prob=params[4])
    learn.lr_find(start_lr=1e-10, end_lr=1e-1)
    learn.model.cuda(1)

    print(learn.model)
    print(summary(learn.model,learn.data.train_ds[:][0].shape[1:]))
    
    start_time = time.time()
    learn.fit_one_cycle(20,max_lr=1.31e-4,wd=5.52e-8,callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=f'Transformer-model-year-{year}-d_model-{params[0]}-n_head-{params[1]}-n_layers-{params[2]}-d_inner-{params[3]}-drop_prob-{params[4]}-percentage-{params[5]}')])
    time_comp = time.time()-start_time

        
    learner_metrics = learn.recorder.metrics_names
    valid_classes = np.argmax(learn.get_preds()[0].numpy(),axis=1)
    
    learn.model.cpu()
    res = {}    
    for idx,metric in enumerate(learner_metrics):
        if metric == 'accuracy':
            res[metric] = accuracy_score(learn.data.valid_ds[:][1],valid_classes)
        elif metric =='f_beta':
            res['F1 score'] = f1_score(learn.data.valid_ds[:][1],valid_classes,average = 'macro')
        elif metric == 'precision':
            res[metric] = precision_score(learn.data.valid_ds[:][1],valid_classes,average = 'macro')
        elif metric == 'recall':
            res[metric] = recall_score(learn.data.valid_ds[:][1],valid_classes,average = 'macro')
    
    res['Time complexity'] = time_comp


    learn.model.cuda(1)
    val_loss = learn.validate()[0]
    
    return res, val_loss


# In[18]:


data = pd.read_csv('SAR_50points_random/50random-dataset-csv.csv')
np.unique(data['class'])


# In[19]:


data = data[data['class'].isin([0,1,2,3,4,5,6,7,13])]
data.loc[data['class']==13,'class'] = 8

dates = data.columns[4:] 
dates = np.array([datetime.strptime(d[:-2], '%Y%m%d') for d in dates[:-1]])
idx = np.where(dates>datetime(2017, 3, 1))[0][:105] + 4 
data_new = data.iloc[:,[0,1,2]+list(idx)] 

data_new.columns = data_new.columns.str.replace('2017','') 
data_new['class'] = data['class']
data_new.head()



# For Transformemr
d_model=[32, 64, 128, 256]
n_head=[1, 2]
n_layers=[4, 5, 6]
d_inner=[64, 128, 256]
drop_prob=[0, 0.25]
percentage = [5, 10, 15, 20, 25, 30]
params = list(product(d_model, n_head, n_layers, d_inner, drop_prob, percentage))




df_res = pd.DataFrame()
month = [('September',105)]

# for y in np.unique(data_new.year)[:-1]:
for y in np.unique(data_new.year)[4:]:
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
            Dodatni deo
            """
            if (p[5]!=30): # Da ne bi imao gresku kad zadas test_size=0 kod split-a.
                # Za train uzimas 1/6 od onih 30% da bi dobio 5% od celog skupa (primera radi). I menjas borjilac u test_size i naziv fajla.
                # Ovde vodis racuna i o broju blokova!!!
                X_trainHELP, X_valHELP, Y_trainHELP, Y_valHELP = train_test_split(X_train, Y_train, stratify=Y_train, test_size=(1-p[5]/30), random_state=42)
                X_train, Y_train = X_trainHELP, Y_trainHELP
            """
            Kraj dodatnog dela
            """

            train_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_train.index])
            # test_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_test.index])
            validation_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_val.index])

            valid_data_learn = valid_data.loc[train_sample_idx, :] # X_train
            valid_data_validate = valid_data.loc[validation_sample_idx, :] #

            X_train = torch.Tensor(valid_data_learn.iloc[:,3:-1].to_numpy())
            Y_train = torch.Tensor(valid_data_learn.iloc[:,-1].to_numpy())

            X_val = torch.Tensor(valid_data_validate.iloc[:,3:-1].to_numpy())
            Y_val = torch.Tensor(valid_data_validate.iloc[:,-1].to_numpy())

            print("TRENING")
            print(np.unique(valid_data_learn.ID_p))
            print(len(np.unique(valid_data_learn.ID_p)))
            print("VALIDACIJA")
            print(np.unique(valid_data_validate.ID_p))
            print(len(np.unique(valid_data_validate.ID_p)))
            print("TESTING")
            print(np.unique(X_test.index))
            print(len(np.unique(X_test.index)))

            
            X_train = X_train.reshape(-1, 1, m[1]) 
            X_val = X_val.reshape(-1, 1, m[1])
            
            data_bunch = create_databunch(X_train, X_val, Y_train.long(), Y_val.long(), bs=1024)
            res, val_loss = train_model(data_bunch, p, y)
            
            res['Validation year'] = int(y)
            res['Month'] = m[0]
            res['Params'] = str(p)
            res['val_loss'] = val_loss 

            df_res = df_res.append(pd.DataFrame(res,index = [0]))
            df_res.to_csv(f'results/hypp_opt_Transformer_2021.csv')

df_res


