import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(0)

from kneed import DataGenerator, KneeLocator
import torch
import torch.nn as nn
from torchsummary import summary
from fastai.basics import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import time
import joblib
import copy

from breizhcrops import TransformerModel

from collections import OrderedDict

from fastai.callbacks.tracker import EarlyStoppingCallback,SaveModelCallback

from torch.utils.data import Dataset, TensorDataset

from pathlib import Path
def list_dir(self, pattern="*"):
    import glob
    return [Path(x) for x in glob.glob(str(self/pattern))]
Path.ls = list_dir

def compute_dimensions(model, input_size, batch_size=-1, device=torch.device('cuda'), dtypes=None):
    '''Generates a input of the specified size and computes the input/output dimensions for every layer'''
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]


    # batch_size of 2 for batchnorm
    x = [ torch.rand(2, *in_size).type(dtype).to(device=device) for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    #print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return summary



class CNN1D(nn.Module):
    '''Create a 1D CNN for the given parameters'''
    class Flatten(nn.Module):
        def forward(self, x):
            x = x.view(x.size()[0], -1)
            return x
    def __init__(self, ninputs = 800, nchannels = 3, nfilters = 64, drop_prob = 0, kernel_size = 4, nclasses=3):
        super().__init__()
        nhead = 32
        self.features = nn.Sequential( 
            # Layer 1
            nn.Conv1d(in_channels=nchannels, out_channels=nfilters, kernel_size=kernel_size),
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(nfilters),
            nn.ReLU(),
            # Layer 2
            nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=kernel_size),
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(nfilters),
            nn.ReLU(),
            # Layer 3
            nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=kernel_size),
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(nfilters),
            nn.ReLU(), 
            # Layer 4
            # nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=kernel_size, padding = int(kernel_size/2)),
            # nn.Dropout(drop_prob),
            # nn.BatchNorm1d(nfilters),
            # nn.ReLU(), 
            # Last Conv layer
            nn.Conv1d(in_channels=nfilters, out_channels=nhead, kernel_size=1),
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(nhead),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            Flatten())

        self.head = nn.Linear(nhead, nclasses)
        
    def forward(self, x):
        return self.head(self.features(x))
    
    
def create_databunch(X_train, X_val, Y_train, Y_val, bs=256):
    x_train, x_val, y_train, y_val = map(torch.tensor, (X_train, X_val, Y_train, Y_val))
        
    train_ds = TensorDataset(x_train.float(), y_train)
    valid_ds = TensorDataset(x_val.float(), y_val)
    
    data = DataBunch.create(train_ds, valid_ds, bs=bs)
    # data = DataLoaders.from_dsets(train_ds, valid_ds, bs=bs)
    
    return data

def create_learner(data, model = None, nfilters = 64, drop_prob = 0, kernel_size = 4):
    if model is None:
        model = CNN1D(nfilters=nfilters, 
                      drop_prob = drop_prob,
                      kernel_size = kernel_size,
                      ninputs = get_dataset_dimensions(data)['ninputs'], 
                      nchannels = get_dataset_dimensions(data)['nchannels'],
                      nclasses = get_dataset_dimensions(data)['nclasses'])
        
    print(f"NFilters={nfilters}, drop_probability = {drop_prob}, kernel_size = {kernel_size}, ninputs={get_dataset_dimensions(data)['ninputs']}, nchannels={get_dataset_dimensions(data)['nchannels']}, nclasses={get_dataset_dimensions(data)['nclasses']}")
    
    average_type = 'macro'
    learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy,FBeta(average=average_type, beta=1),Precision(average=average_type), Recall(average=average_type)])
#     learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy,FBeta(average=average_type, beta=1),Precision(average=average_type), Recall(average=average_type)],callback_fns=[ShowGraph,partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=15)])
          
    print('Number of parameters: ',sum(p.numel() for p in model.parameters()))
          
    return learn

def create_transformer_learner(data, d_model, n_head, n_layers, d_inner, drop_prob, model = None):
    
    model = TransformerModel(input_dim=105, num_classes=9, d_model=d_model, n_head=n_head, n_layers=n_layers, d_inner=d_inner, activation="relu", dropout=drop_prob)
    
    average_type = 'macro'
    learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy,FBeta(average=average_type, beta=1),Precision(average=average_type), Recall(average=average_type)],callback_fns=[ShowGraph,partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=15)])
          
    print('Number of parameters: ',sum(p.numel() for p in model.parameters()))
          
    return learn


def get_dataset_dimensions(db):
    return {
        
        'nchannels': 1,
        'ninputs': db.train_ds[0][0].shape[0],
        'nclasses': np.unique(db.train_ds.tensors[1].numpy()).shape[0]
    }

          
def plot_confusion_matrix(y_true,y_pred,fold,classes,title=None,ylabel='True label',xlabel='Predicted label',normalize=False,cmap=plt.cm.Blues):
    
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.savez(f'results/conf-fold-{fold}',cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]   
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel=ylabel,
           xlabel=xlabel)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()   


def balance(X,Y): 
   
    number_of_samples = np.min(np.bincount(Y))
    class_with_minimum_samples = np.argmin(np.bincount(Y))
    l_x = []
    l_y = []
    for i in range(len(np.unique(Y))):
        if np.bincount(Y)[i]!=number_of_samples:
            indices =random.sample(range(len(Y[Y==i])), number_of_samples)
            l_y.extend(Y[Y==i][indices])
            l_x.extend(X[Y==i][indices])
        else:
            l_x.extend(X[Y==i])
            l_y.extend(Y[Y==i])

    indices = random.sample(range(len(l_y)), len(l_y))
    Y = np.array(l_y)[indices]
    X = np.array(l_x)[indices]
    
    return X,Y

def pareto_front(df_params_sort):
    
    df_pf = pd.DataFrame(columns = ['Params', 'F1 score', 'Time complexity'])
    df_pf = df_pf.append({'Params' : df_params_sort.loc[0,["Params"]][0], 'F1 score' : df_params_sort.loc[0,["F1 score"]][0], 'Time complexity' : df_params_sort.loc[0,["Time complexity"]][0]}, ignore_index = True)

    for i in df_params_sort.index:
        if df_params_sort.loc[i,["F1 score"]][0]>df_pf.loc[df_pf.index[-1],"F1 score"]:
            df_pf = df_pf.append({'Params' : df_params_sort.loc[i,["Params"]][0], 'F1 score' : df_params_sort.loc[i,["F1 score"]][0], 'Time complexity' : df_params_sort.loc[i,["Time complexity"]][0]},ignore_index = True)
    
    return df_pf

def pareto_front_vis(df_params, df_pf):
    
    matplotlib.rcParams.update({'font.size': 16.5})
    plt.figure(figsize=(15,5))
    
    sklj = sns.scatterplot(data=df_params, x="Time complexity", y="F1 score", hue='Params', s=100, legend = False, palette="deep")
    pf = sns.lineplot(data=df_pf, x="Time complexity", y="F1 score", color='green')
    pf = sns.scatterplot(data=df_pf, x="Time complexity", y="F1 score", s=500, legend = False,  color='orange', marker="*")
    
    kneedle = KneeLocator(df_pf['Time complexity'], df_pf['F1 score'], S=1, curve="concave", direction="increasing", online=True)

    if kneedle.knee == None:
        kneedle.knee = df_params.iloc[np.argmax(df_params['F1 score']), :]['Time complexity']

    plt.axvline(kneedle.knee, 0, 1, color = 'k', linestyle='dashed')
    plt.xlabel('Time complexity (sec)')
    plt.grid(True)
    plt.grid(linestyle='dotted')

    return kneedle


def find_best_model(df_res, df_pf, kneedle):

    idx_best_architecture = eval(df_pf.loc[np.where(df_pf['Time complexity']==kneedle.knee)[0],'Params'].values[0])

    best_models = df_res.loc[df_res.Params==str(idx_best_architecture), :]
    best_model = best_models.iloc[best_models['F1 score'].argmax(),:]
    
    return best_model

def train_val_split(data, best_model, month, test_year):
    
    train_idx = data.loc[np.logical_and(data.year!=test_year,data.year!=best_model['Validation year'])].index
    X_train = torch.Tensor(data.loc[train_idx,:].iloc[:,3:-1].to_numpy())
    Y_train = torch.Tensor(data.loc[train_idx,:].iloc[:,-1].to_numpy())

    val_idx = data.loc[data.year==best_model['Validation year']].index
    X_val = torch.Tensor(data.loc[val_idx,:].iloc[:,3:-1].to_numpy())
    Y_val = torch.Tensor(data.loc[val_idx,:].iloc[:,-1].to_numpy())

    X_train = X_train.reshape(-1, 1, month[0][1]) 
    X_val = X_val.reshape(-1, 1, month[0][1])

    return X_train, Y_train, X_val, Y_val

def cnn_learning(data_bunch_create, data_bunch_tl, X_test, Y_test, best_model_cnn, best_model_cnn_2021, res, average_type, iteration):
    
    conf = []
    nconf = []

    if data_bunch_create is not None:
        set_seed(42+iteration)
        learn = create_learner(data_bunch_create, nfilters=eval(best_model_cnn.Params)[0], kernel_size = eval(best_model_cnn.Params)[1], drop_prob = eval(best_model_cnn.Params)[2])
    else:
        set_seed(42+iteration)
        learn = create_learner(data_bunch_tl, nfilters=eval(best_model_cnn.Params)[0], kernel_size = eval(best_model_cnn.Params)[1], drop_prob = eval(best_model_cnn.Params)[2])
        
    learn.load(f'model-year-{best_model_cnn["Validation year"]}-nfilters-{eval(best_model_cnn.Params)[0]}-kernel_size-{eval(best_model_cnn.Params)[1]}-drop_prob-{eval(best_model_cnn.Params)[2]}')
    learn.model.cuda()

    learn.model.eval()
    predictions_cnn = np.argmax(learn.model(torch.Tensor(X_test).cuda()),axis=1)

    res['F1 score CNN original'] = f1_score(Y_test,predictions_cnn,average = average_type)

    conf_NAIVE = confusion_matrix(Y_test, predictions_cnn)
    nconf_NAIVE = conf_NAIVE.astype('float') / conf_NAIVE.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_NAIVE))
    nconf.append(np.ndarray.flatten(nconf_NAIVE))

    
    learn.data = data_bunch_tl
    learn.freeze_to(-1)
    learn.model.train()

    start_time = time.time()
    learn.fit_one_cycle(50,max_lr=1e-3,callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss', name= 'model'), ])
    time_comp = time.time()-start_time

    learn.model.eval()
    predictions_cnn = np.argmax(learn.model(torch.Tensor(X_test).cuda()),axis=1)

    res['F1 score CNN'] = f1_score(Y_test, predictions_cnn, average = average_type)
    res['Accuracy CNN'] = accuracy_score(Y_test, predictions_cnn)
    res['Time complexity CNN'] = time_comp
    
    conf_TL = confusion_matrix(Y_test, predictions_cnn)
    nconf_TL = conf_TL.astype('float') / conf_TL.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_TL))
    nconf.append(np.ndarray.flatten(nconf_TL))



 
    set_seed(42+iteration)
    learn_fs = create_learner(data_bunch_tl, nfilters=eval(best_model_cnn.Params)[0], kernel_size = eval(best_model_cnn.Params)[1], drop_prob = eval(best_model_cnn.Params)[2])
    start_time = time.time()
    learn_fs.fit_one_cycle(50,max_lr=1e-2,callbacks=[SaveModelCallback(learn_fs, every='improvement', monitor='valid_loss', name= 'model_fs')])
    time_comp = time.time()-start_time
    learn_fs.model.eval()
    predictions_cnn = np.argmax(learn_fs.model(torch.Tensor(X_test).cuda()),axis=1)

    res['fsF1 score CNN'] = f1_score(Y_test, predictions_cnn, average = average_type)
    res['fsAccuracy CNN'] = accuracy_score(Y_test, predictions_cnn)
    res['fsTime complexity CNN'] = time_comp

    conf_FS_hist = confusion_matrix(Y_test, predictions_cnn)
    nconf_FS_hist = conf_FS_hist.astype('float') / conf_FS_hist.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_FS_hist))
    nconf.append(np.ndarray.flatten(nconf_FS_hist))



    set_seed(42+iteration)
    learn_fs_2021 = create_learner2(data_bunch_tl, nfilters=eval(best_model_cnn_2021.Params)[0], kernel_size = eval(best_model_cnn_2021.Params)[1], drop_prob = eval(best_model_cnn_2021.Params)[2])
    start_time = time.time()
    learn_fs_2021.fit_one_cycle(50,max_lr=1e-2,callbacks=[SaveModelCallback(learn_fs_2021, every='improvement', monitor='valid_loss', name= 'model_fs_2021')])
    time_comp = time.time()-start_time
    learn_fs_2021.model.eval()
    predictions_cnn = np.argmax(learn_fs_2021.model(torch.Tensor(X_test).cuda()),axis=1)

    res['fsF1 score CNN 2021'] = f1_score(Y_test, predictions_cnn, average = average_type)
    res['fsAccuracy CNN 2021'] = accuracy_score(Y_test, predictions_cnn)
    res['fsTime complexity CNN 2021'] = time_comp

    conf_FS_2021 = confusion_matrix(Y_test, predictions_cnn)
    nconf_FS_2021 = conf_FS_2021.astype('float') / conf_FS_2021.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_FS_2021))
    nconf.append(np.ndarray.flatten(nconf_FS_2021))

    
    print(summary(learn.model,learn.data.train_ds[:][0].shape[1:]))

    return res, conf, nconf


def rf_learning(X_train_percent, Y_train_percent, X_test, Y_test, best_model_rf, best_model_rf_2021, res, average_type, iteration):

    conf = []
    nconf = []


    rf = joblib.load(f'models/RF-year-{best_model_rf["Validation year"]}-n_estimators-{eval(best_model_rf.Params)[0]}-max_depths-{eval(best_model_rf.Params)[1]}-min_samples_leaf-{eval(best_model_rf.Params)[2]}.joblib')
    predictions_rf = rf.predict(np.squeeze(X_test))
    res['F1 score RF original'] = f1_score(Y_test, predictions_rf, average = average_type)

    conf_NAIVE = confusion_matrix(Y_test, predictions_rf)
    nconf_NAIVE = conf_NAIVE.astype('float') / conf_NAIVE.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_NAIVE))
    nconf.append(np.ndarray.flatten(nconf_NAIVE))




    rf.n_estimators += eval(best_model_rf.Params)[0]
    rf.random_state = 42 + iteration
    start_time = time.time()
    rf.fit(np.squeeze(X_train_percent), Y_train_percent)
    time_comp = time.time()-start_time
    predictions_rf = rf.predict(np.squeeze(X_test))

    res['F1 score RF'] = f1_score(Y_test, predictions_rf, average = average_type)
    res['Time complexity RF'] = time_comp

    conf_TL = confusion_matrix(Y_test, predictions_rf)
    nconf_TL = conf_TL.astype('float') / conf_TL.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_TL))
    nconf.append(np.ndarray.flatten(nconf_TL))




    rf_fs = RandomForestClassifier(n_estimators=eval(best_model_rf.Params)[0], max_depth=eval(best_model_rf.Params)[1], min_samples_leaf=eval(best_model_rf.Params)[2], random_state=42+iteration)
    start_time = time.time()
    rf_fs.fit(np.squeeze(X_train_percent), Y_train_percent)
    time_comp = time.time()-start_time
    predictions_rf = rf_fs.predict(np.squeeze(X_test))

    res['fsF1 score RF'] = f1_score(Y_test, predictions_rf, average = average_type)
    res['fsTime complexity RF'] = time_comp

    conf_FS_hist = confusion_matrix(Y_test, predictions_rf)
    nconf_FS_hist = conf_FS_hist.astype('float') / conf_FS_hist.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_FS_hist))
    nconf.append(np.ndarray.flatten(nconf_FS_hist))



    rf_fs_2021 = RandomForestClassifier(n_estimators=eval(best_model_rf_2021.Params)[0], max_depth=eval(best_model_rf_2021.Params)[1], min_samples_leaf=eval(best_model_rf_2021.Params)[2], random_state=42+iteration)
    start_time = time.time()
    rf_fs_2021.fit(np.squeeze(X_train_percent), Y_train_percent)
    time_comp = time.time()-start_time
    predictions_rf = rf_fs_2021.predict(np.squeeze(X_test))

    res['fsF1 score RF 2021'] = f1_score(Y_test, predictions_rf, average = average_type)
    res['fsTime complexity RF 2021'] = time_comp

    conf_FS_2021 = confusion_matrix(Y_test, predictions_rf)
    nconf_FS_2021 = conf_FS_2021.astype('float') / conf_FS_2021.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_FS_2021))
    nconf.append(np.ndarray.flatten(nconf_FS_2021))

    
    return res, conf, nconf

def transformer_learning(data_bunch_create, data_bunch_tl, X_test, Y_test, best_model_transformer, best_model_transformer_2021, res, average_type, iteration):
    
    conf = []
    nconf = []

    if data_bunch_create is not None:
        set_seed(42+iteration)
        learn = create_transformer_learner(data_bunch_create, d_model=eval(best_model_transformer.Params)[0], n_head=eval(best_model_transformer.Params)[1], n_layers=eval(best_model_transformer.Params)[2], d_inner=eval(best_model_transformer.Params)[3], drop_prob=eval(best_model_transformer.Params)[4])
    else:
        set_seed(42+iteration)
        learn = create_transformer_learner(data_bunch_tl, d_model=eval(best_model_transformer.Params)[0], n_head=eval(best_model_transformer.Params)[1], n_layers=eval(best_model_transformer.Params)[2], d_inner=eval(best_model_transformer.Params)[3], drop_prob=eval(best_model_transformer.Params)[4])
    learn.load(f'Transformer-model-year-{best_model_transformer["Validation year"]}-d_model-{eval(best_model_transformer.Params)[0]}-n_head-{eval(best_model_transformer.Params)[1]}-n_layers-{eval(best_model_transformer.Params)[2]}-d_inner-{eval(best_model_transformer.Params)[3]}-drop_prob-{eval(best_model_transformer.Params)[4]}')
    learn.model.cuda()

    learn.model.eval()
    predictions_transformer = np.argmax(learn.model(torch.Tensor(X_test).cuda()),axis=1)

    res['F1 score Transformer original'] = f1_score(Y_test,predictions_transformer,average = average_type)

    conf_NAIVE = confusion_matrix(Y_test, predictions_transformer)
    nconf_NAIVE = conf_NAIVE.astype('float') / conf_NAIVE.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_NAIVE))
    nconf.append(np.ndarray.flatten(nconf_NAIVE))

    
    learn.data = data_bunch_tl
    for name, param in learn.model.named_parameters():
        if name.split('.')[0] != 'outlinear':
            param.requires_grad = False
    learn.model.train()

    start_time = time.time()
    learn.fit_one_cycle(50,max_lr=1.31e-4,wd=5.52e-8,callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss', name= 'model'), ])
    time_comp = time.time()-start_time

    learn.model.eval()
    predictions_transformer = np.argmax(learn.model(torch.Tensor(X_test).cuda()),axis=1)

    res['F1 score Transformer'] = f1_score(Y_test, predictions_transformer, average = average_type)
    res['Accuracy Transformer'] = accuracy_score(Y_test, predictions_transformer)
    res['Time complexity Transformer'] = time_comp
    
    conf_TL = confusion_matrix(Y_test, predictions_transformer)
    nconf_TL = conf_TL.astype('float') / conf_TL.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_TL))
    nconf.append(np.ndarray.flatten(nconf_TL))



 
    set_seed(42+iteration)
    learn_fs = create_transformer_learner(data_bunch_tl, d_model=eval(best_model_transformer.Params)[0], n_head=eval(best_model_transformer.Params)[1], n_layers=eval(best_model_transformer.Params)[2], d_inner=eval(best_model_transformer.Params)[3], drop_prob=eval(best_model_transformer.Params)[4])
    start_time = time.time()
    learn_fs.fit_one_cycle(50,max_lr=1.31e-4,wd=5.52e-8,callbacks=[SaveModelCallback(learn_fs, every='improvement', monitor='valid_loss', name= 'model_fs')])
    time_comp = time.time()-start_time
    learn_fs.model.eval()
    predictions_transformer = np.argmax(learn_fs.model(torch.Tensor(X_test).cuda()),axis=1)

    res['fsF1 score Transformer'] = f1_score(Y_test, predictions_transformer, average = average_type)
    res['fsAccuracy Transformer'] = accuracy_score(Y_test, predictions_transformer)
    res['fsTime complexity Transformer'] = time_comp

    conf_FS_hist = confusion_matrix(Y_test, predictions_transformer)
    nconf_FS_hist = conf_FS_hist.astype('float') / conf_FS_hist.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_FS_hist))
    nconf.append(np.ndarray.flatten(nconf_FS_hist))



    set_seed(42+iteration)
    learn_fs_2021 = create_transformer_learner(data_bunch_tl, d_model=eval(best_model_transformer_2021.Params)[0], n_head=eval(best_model_transformer_2021.Params)[1], n_layers=eval(best_model_transformer_2021.Params)[2], d_inner=eval(best_model_transformer_2021.Params)[3], drop_prob=eval(best_model_transformer_2021.Params)[4])
    start_time = time.time()
    learn_fs_2021.fit_one_cycle(50,max_lr=1.31e-4,wd=5.52e-8,callbacks=[SaveModelCallback(learn_fs_2021, every='improvement', monitor='valid_loss', name= 'model_fs_2021')])
    time_comp = time.time()-start_time
    learn_fs_2021.model.eval()
    predictions_transformer = np.argmax(learn_fs_2021.model(torch.Tensor(X_test).cuda()),axis=1)

    res['fsF1 score Transformer 2021'] = f1_score(Y_test, predictions_transformer, average = average_type)
    res['fsAccuracy Transformer 2021'] = accuracy_score(Y_test, predictions_transformer)
    res['fsTime complexity Transformer 2021'] = time_comp

    conf_FS_2021 = confusion_matrix(Y_test, predictions_transformer)
    nconf_FS_2021 = conf_FS_2021.astype('float') / conf_FS_2021.sum(axis=1)[:, np.newaxis]

    conf.append(np.ndarray.flatten(conf_FS_2021))
    nconf.append(np.ndarray.flatten(nconf_FS_2021))

    
    print(summary(learn.model,learn.data.train_ds[:][0].shape[1:]))

    return res, conf, nconf



def sk_fold(n_folds, data_new, best_model_cnn, best_model_cnn_2021, best_model_rf, best_model_rf_2021, res, df_res, average_type, iteration, month, points_of_interest, conf_list, nconf_list):
    
#     valid_data = data_new.loc[data_new.year==best_model_cnn['Validation year']]
        
    valid_data = data_new.loc[data_new.year==2021]
    valid_per_field_all = valid_data.groupby('ID_p').mean()

    valid_data_learn = copy.deepcopy(valid_data)
    valid_data_validate = copy.deepcopy(valid_data)
    
    skf = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)

    for fold, (_, test_fold_per_field_idx) in enumerate(skf.split(valid_per_field_all.iloc[:,:-1], valid_per_field_all.iloc[:,-1])):
        print(f"Fold {fold}:")

        test_fold_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==valid_per_field_all.iloc[t].name,:].index for t in test_fold_per_field_idx])

        valid_data_learn = valid_data.loc[test_fold_sample_idx, :]
        valid_data_validate = valid_data.drop(index=test_fold_sample_idx)

        valid_data_learn_fields = valid_data_learn.groupby('ID_p').mean()
        valid_data_learn_fields_classes = np.unique(valid_data_learn_fields['class'])

        for step in np.round(np.arange(0.16666666666666669, 1+0.00000000000001, 0.16666666666666669),3)[points_of_interest]:
        # for step in np.arange(0.05, 1+0.0001, 0.05)[points_of_interest]:

            valid_data_learn_small = pd.DataFrame([])

            for tc in valid_data_learn_fields_classes.astype(int):
                random.seed(iteration)
                test_fold_per_field_idx = list(valid_data_learn_fields.loc[valid_data_learn_fields['class']==tc,:].index)
                if step == 1:
                    test_per_field_step_idx = test_fold_per_field_idx
                else:
                    test_per_field_step_idx = random.sample(test_fold_per_field_idx, int(np.ceil(len(test_fold_per_field_idx)*step)))

                test_sample_step_idx = np.hstack([valid_data_learn.loc[valid_data_learn['ID_p']==t,:].index for t in test_per_field_step_idx])
                # valid_data_learn_small = Step percentage of 20% of validation dataset. 20% is coiming from 5-fold split.
                valid_data_learn_small = pd.concat([valid_data_learn_small, valid_data_learn.loc[test_sample_step_idx, :]])

            X_test = torch.Tensor(valid_data_validate.iloc[:,3:-1].to_numpy())
            Y_test = torch.Tensor(valid_data_validate.iloc[:,-1].to_numpy())
            X_test = X_test.reshape(-1, 1, month[0][1]) 

            X_train_percent = torch.Tensor(valid_data_learn_small.iloc[:,3:-1].to_numpy())
            Y_train_percent = torch.Tensor(valid_data_learn_small.iloc[:,-1].to_numpy())
            X_train_percent = X_train_percent.reshape(-1, 1, month[0][1]) 
                        
            data_bunch_tl = create_databunch(X_train_percent, X_train_percent, Y_train_percent.long(), Y_train_percent.long(), bs=512)
            
            res['Fold'] = fold
            res['Iteration'] = iteration
            res['Training ratio'] = step*100/skf.n_splits

            #### CNN ###
            if best_model_cnn is not None:
                res, conf, nconf = cnn_learning(None, data_bunch_tl, X_test, Y_test, best_model_cnn, best_model_cnn_2021, res, average_type)
                
            #### RF ###
            
            if best_model_rf is not None:
                res, _, _ = rf_learning(X_train_percent, Y_train_percent, X_test, Y_test, best_model_rf, best_model_rf_2021, res, average_type)

            conf_list.append(np.ndarray.flatten(conf))
            nconf_list.append(np.ndarray.flatten(nconf))
            
            #np.savez('conf_matrices_validation', conf_list, nconf_list)
            
            df_res = df_res.append(pd.DataFrame(res,index = [0]))
            df_res.to_csv('SK_results.csv')

    return df_res, conf_list, nconf_list


def initial(data_new, best_model_cnn, best_model_cnn_2021, best_model_rf, best_model_rf_2021, res, df_res, average_type, iteration, month, points_of_interest, conf_list, nconf_list):
            
    valid_data = data_new.loc[data_new.year==2021]
    valid_per_field_all = valid_data.groupby('ID_p').mean()

    # valid_data_learn = copy.deepcopy(valid_data)
    # valid_data_validate = copy.deepcopy(valid_data)
    
    X = valid_per_field_all.iloc[:,:-1]
    Y = valid_per_field_all.iloc[:,-1]
    X_prim, X_test, Y_prim, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.6, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_prim, Y_prim, stratify=Y_prim, test_size=0.25, random_state=42)
    train_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_train.index])
    test_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==t,:].index for t in  X_test.index])

    valid_data_learn = valid_data.loc[train_sample_idx, :] # AKA valid_data_learn = valid_data.loc[test_fold_sample_idx, :]
    valid_data_validate = valid_data.loc[test_sample_idx, :] # AKA valid_data_validate = valid_data.drop(index=test_fold_sample_idx)

    print(train_sample_idx)

    # for fold, (_, test_fold_per_field_idx) in enumerate(skf.split(valid_per_field_all.iloc[:,:-1], valid_per_field_all.iloc[:,-1])):
    #     print(f"Fold {fold}:")

    #     test_fold_sample_idx = np.hstack([valid_data.loc[valid_data['ID_p']==valid_per_field_all.iloc[t].name,:].index for t in test_fold_per_field_idx])

    #     valid_data_learn = valid_data.loc[test_fold_sample_idx, :]
    #     valid_data_validate = valid_data.drop(index=test_fold_sample_idx)

    valid_data_learn_fields = valid_data_learn.groupby('ID_p').mean()
    valid_data_learn_fields_classes = np.unique(valid_data_learn_fields['class'])

    # for step in np.round(np.arange(0.16666666666666669, 1+0.00000000000001, 0.16666666666666669),3)[points_of_interest]: # 5%,10%,15%,20%,25%,30%
    for step in np.round(np.arange(0.0333333333333333, 1+0.00000000000001, 0.0333333333333333333),3)[points_of_interest]: # 1%,2%,3%,...,29%,30%

        valid_data_learn_small = pd.DataFrame([])

        for tc in valid_data_learn_fields_classes.astype(int):
            random.seed(iteration)
            test_fold_per_field_idx = list(valid_data_learn_fields.loc[valid_data_learn_fields['class']==tc,:].index)
            if step == 1:
                test_per_field_step_idx = test_fold_per_field_idx
            else:
                test_per_field_step_idx = random.sample(test_fold_per_field_idx, int(np.ceil(len(test_fold_per_field_idx)*step)))

            test_sample_step_idx = np.hstack([valid_data_learn.loc[valid_data_learn['ID_p']==t,:].index for t in test_per_field_step_idx])
            # valid_data_learn_small = Step percentage of 20% of validation dataset. 20% is coiming from 5-fold split.
            valid_data_learn_small = pd.concat([valid_data_learn_small, valid_data_learn.loc[test_sample_step_idx, :]])

        X_test = torch.Tensor(valid_data_validate.iloc[:,3:-1].to_numpy())
        Y_test = torch.Tensor(valid_data_validate.iloc[:,-1].to_numpy())
        X_test = X_test.reshape(-1, 1, month[0][1]) 

        X_train_percent = torch.Tensor(valid_data_learn_small.iloc[:,3:-1].to_numpy())
        Y_train_percent = torch.Tensor(valid_data_learn_small.iloc[:,-1].to_numpy())
        X_train_percent = X_train_percent.reshape(-1, 1, month[0][1]) 
                    
        data_bunch_tl = create_databunch(X_train_percent, X_train_percent, Y_train_percent.long(), Y_train_percent.long(), bs=512)
        
        # res['Fold'] = fold
        res['Iteration'] = iteration
        res['Training ratio'] = int(np.round(step*30))

        #### CNN ###
        if best_model_cnn is not None:
            res, conf, nconf = cnn_learning(None, data_bunch_tl, X_test, Y_test, best_model_cnn, best_model_cnn_2021, res, average_type, iteration)               

        conf_list.append(np.ndarray.flatten(conf))
        nconf_list.append(np.ndarray.flatten(nconf))

        #### RF ###
        
        if best_model_rf is not None:
            print(best_model_rf)
            res, conf, nconf = rf_learning(X_train_percent, Y_train_percent, X_test, Y_test, best_model_rf, best_model_rf_2021, res, average_type, iteration)
        
        conf_list.append(np.ndarray.flatten(conf))
        nconf_list.append(np.ndarray.flatten(nconf))
        
        #np.savez('conf_matrices_validation', conf_list, nconf_list)
        
        df_res = df_res.append(pd.DataFrame(res,index = [0]))
        df_res.to_csv('Initial_results_each1pct.csv')

    return df_res, conf_list, nconf_list

def find_best_rf(data_path):
    df_res = pd.read_csv(data_path)
    df_res.drop(df_res.columns[0], axis=1, inplace=True)

    drop_10_50_None = [list(np.where(df_res.Params == i)[0]) for i in df_res.Params if eval(i)[0] < 100 or eval(i)[1]==None]
    flat_list = [item for sublist in drop_10_50_None for item in sublist]
    flat_list = np.unique(flat_list)

    df_res.drop(index=flat_list, inplace=True)

    df_params = df_res.groupby(by=["Params"]).mean()
    df_params.reset_index(inplace=True) 

    df_params_sort = df_params.sort_values(by=["Time complexity"]) 
    df_params_sort.reset_index(inplace=True, drop=True) 

    df_pf = pareto_front(df_params_sort)

    kneedle = pareto_front_vis(df_params, df_pf)

    best_model_rf = find_best_model(df_res, df_pf, kneedle)
    plt.savefig('best_model_rf-mean.pdf')

    return best_model_rf


def find_best_cnn(data_path):
    df_res = pd.read_csv(data_path)
    df_res.drop(df_res.columns[0], axis=1, inplace=True)

    df_res['Params'] = [df_res['Params'][p][:-1] + f", {df_res['Number of blocks'][p]}" + df_res['Params'][p][-1:] for p in range(df_res.shape[0])]

    df_params = df_res.groupby(by=['Params']).mean()
    df_params.reset_index(inplace=True) 

    df_params_sort = df_params.sort_values(by=["Time complexity"]) 
    df_params_sort.reset_index(inplace=True, drop=True) 
        
    df_pf = pareto_front(df_params_sort)
                
    kneedle = pareto_front_vis(df_params, df_pf)

    best_model_cnn = find_best_model(df_res, df_pf, kneedle)
    plt.savefig('best_model_cnn-mean.pdf')
    return best_model_cnn

def find_best_transformer(data_path):
    df_res = pd.read_csv(data_path)
    df_res.drop(df_res.columns[0], axis=1, inplace=True)

    df_params = df_res.groupby(by=['Params']).mean()
    df_params.reset_index(inplace=True) 

    df_params_sort = df_params.sort_values(by=["Time complexity"]) 
    df_params_sort.reset_index(inplace=True, drop=True) 
        
    df_pf = pareto_front(df_params_sort)
                
    kneedle = pareto_front_vis(df_params, df_pf)

    best_model_transformer = find_best_model(df_res, df_pf, kneedle)
    plt.savefig('best_model_transformer-mean.pdf')
    return best_model_transformer

def set_seed(x): 
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)


# ---------------------------------------------------------------------------------------------

def find_best_rf2(data_path):
    df_res = pd.read_csv(data_path)
    df_res.drop(df_res.columns[0], axis=1, inplace=True)

    best_model_rf = []

    for i, row in df_res.iterrows():
        df_res.at[i,'percentage'] =  df_res["Params"][i][1:-1].split(",")[-1][1:] # glupa podela, ali ima neki visak "space"-a kad uradi split

    procenti = ["5", "10", "15", "20", "25", "30"]
    for i in procenti:
        df_res_i = df_res.loc[df_res["percentage"] == i]
        df_params = df_res_i
        
        df_params_sort = df_params.sort_values(by=["Time complexity"]) 
        df_params_sort.reset_index(inplace=True, drop=True) 

        df_pf = pareto_front(df_params_sort)

        kneedle = pareto_front_vis(df_params, df_pf)

        best_model_rf.append(find_best_model(df_res, df_pf, kneedle))
        plt.savefig(f'best_model_rf-mean_{i}.pdf')

    return best_model_rf

def find_best_cnn2(data_path):
    df_res = pd.read_csv(data_path)
    df_res.drop(df_res.columns[0], axis=1, inplace=True)

    best_model_cnn = []

    for i, row in df_res.iterrows():
        df_res.at[i,'percentage'] =  df_res["Params"][i][1:-1].split(",")[-1][1:] # glupa podela, ali ima neki visak "space"-a kad uradi sp

    df_res['Params'] = [df_res['Params'][p][:-1] + f", {df_res['Number of blocks'][p]}" + df_res['Params'][p][-1:] for p in range(df_res.shape[0])]

    procenti = ["5", "10", "15", "20", "25", "30"]
    for i in procenti:
        df_res_i = df_res.loc[df_res["percentage"] == i]
        df_params = df_res_i

        df_params_sort = df_params.sort_values(by=["Time complexity"]) 
        df_params_sort.reset_index(inplace=True, drop=True) 

        df_pf = pareto_front(df_params_sort)

        kneedle = pareto_front_vis(df_params, df_pf)

        best_model_cnn.append(find_best_model(df_res, df_pf, kneedle))
        plt.savefig(f'best_model_cnn-mean_{i}.pdf')

    return best_model_cnn

def find_best_transformer2(data_path):
    df_res = pd.read_csv(data_path)
    df_res.drop(df_res.columns[0], axis=1, inplace=True)

    best_model_transformer = []

    for i, row in df_res.iterrows():
        df_res.at[i,'percentage'] =  df_res["Params"][i][1:-1].split(",")[-1][1:] # glupa podela, ali ima neki visak "space"-a kad uradi split

    procenti = ["5", "10", "15", "20", "25", "30"]
    for i in procenti:
        
        df_res_i = df_res.loc[df_res["percentage"] == i]
        df_params = df_res_i
        
        df_params_sort = df_params.sort_values(by=["Time complexity"]) 
        df_params_sort.reset_index(inplace=True, drop=True) 
        
        df_pf = pareto_front(df_params_sort)

        kneedle = pareto_front_vis(df_params, df_pf)

        best_model_transformer.append(find_best_model(df_res, df_pf, kneedle))
        plt.savefig(f'best_model_transformer-mean_{i}.pdf')


    return best_model_transformer


def initial2(data_new, best_model_cnn, best_model_cnn_2021, best_model_rf, best_model_rf_2021, best_model_transformer, best_model_transformer_2021, res, df_res, average_type, iteration, month, points_of_interest, conf_list, nconf_list):
            
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
            # valid_data_learn_small = Step percentage of 20% of validation dataset. 20% is coiming from 5-fold split.
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

        #### CNN ###
        if best_model_cnn is not None:
            res, conf, nconf = cnn_learning(None, data_bunch_tl, X_test, Y_test, best_model_cnn, best_model_cnn_2021[idx], res, average_type, iteration)               

        # conf_list.append(np.ndarray.flatten(conf))
        # nconf_list.append(np.ndarray.flatten(nconf))        

        # conf_list.append(conf)
        # nconf_list.append(nconf)  

        #### RF ###
        if best_model_rf is not None:
            print(best_model_rf)
            res, conf, nconf = rf_learning(X_train_percent, Y_train_percent, X_test, Y_test, best_model_rf, best_model_rf_2021[idx], res, average_type, iteration)
        
        # conf_list.append(np.ndarray.flatten(conf))
        # nconf_list.append(np.ndarray.flatten(nconf))

        # conf_list.append(conf)
        # nconf_list.append(nconf)


        #### Transformer ###
        if best_model_transformer is not None:
            res, conf, nconf = transformer_learning(None, data_bunch_tl, X_test, Y_test, best_model_transformer, best_model_transformer_2021[idx], res, average_type, iteration)               

        # conf_list.append(np.ndarray.flatten(conf))
        # nconf_list.append(np.ndarray.flatten(nconf))        

        conf_list.append(conf)
        nconf_list.append(nconf)  




        
        np.savez('Conf_matrices_validation-Transformer', conf_list, nconf_list)
        
        df_res = df_res.append(pd.DataFrame(res,index = [0]))
        df_res.to_csv('Initial2_results_5-10-15-20-25-30_Transformer.csv')

    return df_res, conf_list, nconf_list

def create_learner2(data, model = None, nfilters = 64, drop_prob = 0, kernel_size = 4):
    if model is None:
        model = CNN1D_2(nfilters=nfilters, 
                      drop_prob = drop_prob,
                      kernel_size = kernel_size,
                      ninputs = get_dataset_dimensions(data)['ninputs'], 
                      nchannels = get_dataset_dimensions(data)['nchannels'],
                      nclasses = get_dataset_dimensions(data)['nclasses'])
        
    print(f"NFilters={nfilters}, drop_probability = {drop_prob}, kernel_size = {kernel_size}, ninputs={get_dataset_dimensions(data)['ninputs']}, nchannels={get_dataset_dimensions(data)['nchannels']}, nclasses={get_dataset_dimensions(data)['nclasses']}")
    
    average_type = 'macro'
    learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy,FBeta(average=average_type, beta=1),Precision(average=average_type), Recall(average=average_type)])
#     learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy,FBeta(average=average_type, beta=1),Precision(average=average_type), Recall(average=average_type)],callback_fns=[ShowGraph,partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=15)])
          
    print('Number of parameters: ',sum(p.numel() for p in model.parameters()))
          
    return learn

class CNN1D_2(nn.Module):
    '''Create a 1D CNN for the given parameters'''
    class Flatten(nn.Module):
        def forward(self, x):
            x = x.view(x.size()[0], -1)
            return x
    def __init__(self, ninputs = 800, nchannels = 3, nfilters = 64, drop_prob = 0, kernel_size = 4, nclasses=3):
        super().__init__()
        nhead = 32
        self.features = nn.Sequential( 
            # Layer 1
            nn.Conv1d(in_channels=nchannels, out_channels=nfilters, kernel_size=kernel_size),
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(nfilters),
            nn.ReLU(),
            # Layer 2
            nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=kernel_size),
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(nfilters),
            nn.ReLU(),
            # Layer 3
            # nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=kernel_size),
            # nn.Dropout(drop_prob),
            # nn.BatchNorm1d(nfilters),
            # nn.ReLU(), 
            # Layer 4
            # nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=kernel_size, padding = int(kernel_size/2)),
            # nn.Dropout(drop_prob),
            # nn.BatchNorm1d(nfilters),
            # nn.ReLU(), 
            # Last Conv layer
            nn.Conv1d(in_channels=nfilters, out_channels=nhead, kernel_size=1),
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(nhead),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            Flatten())

        self.head = nn.Linear(nhead, nclasses)
        
    def forward(self, x):
        return self.head(self.features(x))
    
    