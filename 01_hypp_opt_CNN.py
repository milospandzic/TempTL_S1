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


# In[11]:


def set_seed(x=42): 
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)


# In[17]:


def train_model(data, params, year):
    
    set_seed()
    learn = create_learner(data, nfilters=params[0], kernel_size = params[1], drop_prob=params[2])
    learn.lr_find(start_lr=1e-10, end_lr=1e-1)
    learn.model.cuda(1)

#     learn.recorder.plot()
    print(learn.model)
    print(summary(learn.model,learn.data.train_ds[:][0].shape[1:]))
    
    start_time = time.time()
    learn.fit_one_cycle(20,max_lr=1e-3,callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=f'model-year-{year}-nfilters-{params[0]}-kernel_size-{params[1]}-drop_prob-{params[2]}-layer_blocks-2-percentage-{params[3]}')])
    time_comp = time.time()-start_time
    # learn.save(f'model-year-{year}-nfilters-{params[0]}-kernel_size-{params[1]}-drop_prob-{params[2]}-layer_blocks-4-percentage-{params[3]}')
    
#     learn.recorder.plot_losses();
#     plt.savefig(f'results/losses-year-{year}-nfilters-{nfilters}-drop_prob-{drop_prob}-kernel_size-{kernel_size}.png', dpi=300)
#     plt.close()
        
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

#     plot_confusion_matrix(learn.data.valid_ds[:][1],valid_classes,year,np.unique(valid_classes),title=res['accuracy'],ylabel='True label',xlabel='Predicted label',normalize=False,cmap=plt.cm.Blues)
#     plot_confusion_matrix(learn.data.valid_ds[:][1],valid_classes,year,np.unique(valid_classes),title=res['accuracy'],ylabel='True label',xlabel='Predicted label',normalize=True,cmap=plt.cm.Blues)
#     plt.savefig(f'results/conf-year-{year}-nfilters-{nfilters}-drop_prob-{drop_prob}-kernel_size-{kernel_size}.png', dpi=300)
#     plt.close()
    
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


# In[20]:


nfilters = [4, 8, 16, 32]
kernel_sizes = [4, 8, 16, 32]
drop_probs = [0.25, 0.50, 0.75]
percentage = [5, 10, 15, 20, 25, 30]

params = list(product(nfilters, kernel_sizes, drop_probs, percentage))


# In[21]:


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
            if (p[3]!=30): # Da ne bi imao gresku kad zadas test_size=0 kod split-a.
                # Za train uzimas 1/6 od onih 30% da bi dobio 5% od celog skupa (primera radi). I menjas borjilac u test_size i naziv fajla.
                # Ovde vodis racuna i o broju blokova!!!
                X_trainHELP, X_valHELP, Y_trainHELP, Y_valHELP = train_test_split(X_train, Y_train, stratify=Y_train, test_size=(1-p[3]/30), random_state=42)
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

            # train_idx = data_new.loc[np.logical_and(data_new.year!=2021,data_new.year!=y)].index
            # X_train = torch.Tensor(data_new.loc[train_idx,:].iloc[:,3:-1].to_numpy())
            # Y_train = torch.Tensor(data_new.loc[train_idx,:].iloc[:,-1].to_numpy())
            
            # val_idx = data_new.loc[data_new.year==y].index
            # X_val = torch.Tensor(data_new.loc[val_idx,:].iloc[:,3:-1].to_numpy())
            # Y_val = torch.Tensor(data_new.loc[val_idx,:].iloc[:,-1].to_numpy())
            
            X_train = X_train.reshape(-1, 1, m[1]) 
            X_val = X_val.reshape(-1, 1, m[1])
            
            data_bunch = create_databunch(X_train, X_val, Y_train.long(), Y_val.long(), bs=1024)
            res, val_loss = train_model(data_bunch, p, y)
            
            res['Validation year'] = int(y)
            res['Month'] = m[0]
            res['Params'] = str(p)
            res['val_loss'] = val_loss 

            df_res = df_res.append(pd.DataFrame(res,index = [0]))
            df_res.to_csv(f'results/hypp_opt_CNN_2_layer_blocks2021.csv')


# In[ ]:


df_res


# In[ ]:


# for y in np.unique(data_new.year)[:-1]:
#     print(y)
#     df_res_year = df_res.loc[df_res['Validation year']==y,:]
#     table = df_res_year.pivot_table(values='F1 score', index='Params', columns=['Month'])
#     table = table.reindex(columns=df_res_year['Month'].unique())
#     display(table.style.highlight_max(color='lightgreen',axis=0))


# In[ ]:


# nfilers x kernel_size = ? * ?
# 4 x 4 = 4 * 96 
# 4 x 16 = 4 * 60
# 16 x 16 = 32 * 60
# 32 x 8 = 32 * 84
# 32 x 32 = 32 * 12


# In[ ]:




