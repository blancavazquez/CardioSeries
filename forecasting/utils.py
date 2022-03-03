import os
import sys
import json
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from math import sqrt
from time import time
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.utils import class_weight

#Loading pytorch_lightning
import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score, \
                            roc_auc_score, auc,roc_curve,average_precision_score, \
                            accuracy_score,f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
warnings.filterwarnings('ignore')

def get_df(imputation):
    """
    Recuerda: se volvió a ejecutar "imputer", code in 
    "/home/blanca/anaconda3/envs/time_series_forecasting/impute/"
    
    Los nuevos set están almacenados dentro del folder de imputer
    """
    path_data = "impute/data/"
    train=pd.read_csv(path_data+"train_"+imputation+".csv")
    val=pd.read_csv(path_data+"val_"+imputation+".csv")

    #train = train[:6144]
    #val = val[:6144]

    return train, val

class LoadingDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target):
        # Initialization
        super(LoadingDataset, self).__init__()  
        self.X_input = X_input #(128, 24, 8)
        self.X_target = X_target #(128, 24, 8)
        
    def __len__(self):
        # Retorna el número de muestras en el dataset
        return (self.X_input).shape[0]

    def __getitem__(self, idx): 
        #Loads and returns a sample from the dataset at the given index idx.
        #X_input: (24, 1, 8), X_target: (24, 1, 8)
        #return (self.X_input[idx,:,np.newaxis], self.X_target[idx,:,np.newaxis]) #convertir una matriz 1D hacia vector columna
        return (self.X_input[idx],self.X_target[idx])

def loading_dataloader(past_seqs,targets_seqs,batch_size,dataset):
    dataset_torch = LoadingDataset(past_seqs,targets_seqs) #xtrain, ytrain #23540

    if dataset =="Train":
        loader = DataLoader(dataset_torch, batch_size=batch_size,shuffle=True, num_workers=1)
    else:
        loader = DataLoader(dataset_torch, batch_size=batch_size,shuffle=False, num_workers=1)

    print("dataset_torch:",len(dataset_torch),"loader:",len(loader))
    return loader


def loading_dataloader_conditional(past,expected,batch_size,seq_length,conditions,dataset):
    print("past:", past.shape,"expected:",expected.shape,"conditions:", conditions.shape)
    pd_cond = pd.DataFrame(conditions[['pid','age','gender']])    

    grp_by_cond = pd_cond.groupby(by=['pid'])
    grp_by_data = past.groupby(by=['pid'])

    groups_cond = list(grp_by_cond.groups)
    groups_data= list(grp_by_data.groups)

    full_groups_data= [grp for grp in groups_data if grp_by_data.get_group(grp).shape[0]>=24]

    dataset_torch = LoadingDataset_conditional(X_input= past, X_target=expected, 
                                               conditions= grp_by_cond,groups=full_groups_data,
                                               grp_by=grp_by_data)

    if dataset =="Train":
        loader = DataLoader(dataset_torch, batch_size=batch_size,shuffle=True, num_workers=1)
    else:
        loader = DataLoader(dataset_torch, batch_size=batch_size,shuffle=False, num_workers=1)

    return loader

def df_to_np(df):
    arr = np.array(df)
    arr = pad_arr(arr)
    return arr

class LoadingDataset_conditional(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target, conditions, groups, grp_by):
        # Initialization
        super(LoadingDataset_conditional, self).__init__()  
        self.X_input = X_input #(128, 24, 8)
        self.X_target = X_target #(128, 24, 8)
        self.cond = conditions
        self.groups = groups
        self.grp_by = grp_by
                
    def __len__(self):
        # Retorna el número de muestras en el dataset
        return len(self.groups)

    def __getitem__(self, idx): 
        #Loads and returns a sample from the dataset at the given index idx.
        #X_input: (24, 1, 8), X_target: (24, 1, 8)

        pid_patient = self.groups[idx]

        df = self.grp_by.get_group(pid_patient) #get all features for each patient

        df_conditions = self.cond.get_group(pid_patient)

        df_conditions = df_conditions[['age','gender']]

        feature_list = ['arterial_bp_mean','respiratory_rate', 
                        'diastolic_bp','spo2', 'o2_saturation','heart_rate', 
                        'systolic_bp', 'temperature']

        targets_in = self.X_input[feature_list] #(24,8)
        targets_in = np.array(targets_in)

        targets_in = targets_in.reshape(-1,24,8)

        targets_out = self.X_target[feature_list] #(24,8)
        targets_out = np.array(targets_out)
        targets_out = targets_out.reshape(-1,24,8)

        conditions = np.array(df_conditions) #sex, age
        conditions = conditions.flatten()

        targets_in = torch.tensor(targets_in, dtype=torch.float) #torch.Size([24, 8])
        targets_out = torch.tensor(targets_out, dtype=torch.float) #torch.Size([24, 8])
        conditions = torch.tensor(conditions,dtype=torch.float) #[1,2]
        
        #print("targets_in:", targets_in.shape, "targets_out:",targets_out.shape, "conditions:",conditions.shape)
        return (targets_in, targets_out,conditions)


def scale_data(train, vali,next_train,next_vali,path):
    "Scaling data and saving scaler"

    train_r = train.reshape(-1, train.shape[1]*train.shape[2])
    vali_r = vali.reshape(-1, train.shape[1]*train.shape[2])
    next_train_r = next_train.reshape(-1, train.shape[1]*train.shape[2])
    next_vali_r = next_vali.reshape(-1, train.shape[1]*train.shape[2])
    
    scale_range=(0, 1)
    scaler = MinMaxScaler(feature_range=scale_range).fit(np.vstack([train_r]))

    # scale everything
    #escalando_train = scaler.transform(train_r)
    scaled_train = scaler.transform(train_r).reshape(-1, train.shape[1], train.shape[2])
    scaled_vali = scaler.transform(vali_r).reshape(-1, train.shape[1], train.shape[2])
    scaled_next_train= scaler.transform(next_train_r).reshape(-1, train.shape[1], train.shape[2])
    scaled_next_vali= scaler.transform(next_vali_r).reshape(-1, train.shape[1], train.shape[2])

    #Saving scaler
    pickle.dump(scaler, open(path+'data/'+'scaler', 'wb'))

    return scaled_train, scaled_vali, scaled_next_train,scaled_next_vali

def get_values_per_hour(data,seq_length,feature_list):
    patients = set(data.pid)
    X = np.empty(shape=(len(patients), len(feature_list)*seq_length))
    i=0
    for pat in patients:
        values = data[data.pid == pat]
        timeseries = values[feature_list]
        if timeseries.shape[0]!=23:
            timeseries = timeseries.to_numpy().reshape(len(feature_list)*seq_length)
            X[i, :] = timeseries
            i+=1
    return X

def loading_data_secuencial(feature_list,seq_length,path):
    train,val = get_df("iterative")

    Xtrain = train[(train.offset>=0)&(train.offset<=23)]
    ytrain= train[(train.offset>=24)&(train.offset<=47)]

    Xval = val[(val.offset>=0)&(val.offset<=23)]
    yval= val[(val.offset>=24)&(val.offset<=47)]

    # train_bkp = Xtrain.offset.values
    # val_bkp = Xval.offset.values

    Xfirst_24hrs_train = get_values_per_hour(Xtrain,seq_length,feature_list)
    ynext_24hrs_train = get_values_per_hour(ytrain,seq_length,feature_list)
    Xfirst_24hrs_val = get_values_per_hour(Xval,seq_length,feature_list)
    ynext_24hrs_val = get_values_per_hour(yval,seq_length,feature_list)
    
    Xtrain= np.around(np.reshape(Xfirst_24hrs_train, (-1, seq_length, len(feature_list))),3)
    ytrain = np.around(np.reshape(ynext_24hrs_train, (-1, seq_length, len(feature_list))),3)
    Xval= np.around(np.reshape(Xfirst_24hrs_val, (-1, seq_length, len(feature_list))),3)
    yval= np.around(np.reshape(ynext_24hrs_val, (-1, seq_length, len(feature_list))),3)

    ##-------Scaling data-------#
    Xtrain,Xval,ytrain,yval = scale_data(Xtrain,Xval,ytrain,yval,path)

    Xtrain = Xtrain.reshape(-1,Xtrain.shape[1]*Xtrain.shape[2])

    ytrain = ytrain.reshape(-1,ytrain.shape[1]*ytrain.shape[2])

    Xval = Xval.reshape(-1,Xval.shape[1]*Xval.shape[2])

    yval = yval.reshape(-1,yval.shape[1]*yval.shape[2])
    
    return Xtrain,ytrain,Xval,yval


def get_data_plus_pid(seq_length,feature_list,path):
    "Function to extract EHR data + PID (used for CGAN)"
    train,val = get_df("iterative")

    Xtrain = train[(train.offset>=0)&(train.offset<=23)]
    ytrain= train[(train.offset>=24)&(train.offset<=47)]

    Xval = val[(val.offset>=0)&(val.offset<=23)]
    yval= val[(val.offset>=24)&(val.offset<=47)]

    Xfirst_24hrs_train = get_values_per_hour(Xtrain,seq_length,feature_list)
    ynext_24hrs_train = get_values_per_hour(ytrain,seq_length,feature_list)
    Xfirst_24hrs_val = get_values_per_hour(Xval,seq_length,feature_list)
    ynext_24hrs_val = get_values_per_hour(yval,seq_length,feature_list)

    Xtrain= np.around(np.reshape(Xfirst_24hrs_train, (-1*seq_length,len(feature_list))),3)
    ytrain = np.around(np.reshape(ynext_24hrs_train, (-1*seq_length,len(feature_list))),3)
    Xval= np.around(np.reshape(Xfirst_24hrs_val, (-1*seq_length,len(feature_list))),3)
    yval= np.around(np.reshape(ynext_24hrs_val, (-1*seq_length,len(feature_list))),3)

    Xtrain = pd.DataFrame(Xtrain,columns=feature_list)
    ytrain = pd.DataFrame(ytrain,columns=feature_list)
    Xval = pd.DataFrame(Xval,columns=feature_list)
    yval = pd.DataFrame(yval,columns=feature_list)
    
    return Xtrain,ytrain,Xval,yval


def create_folders(net):
    """Function for creating folders for saving results"""
    if net =="vae":
        if not os.path.isdir('experimentos/vae/'):
            os.mkdir('experimentos/vae/')
        if not os.path.isdir('experimentos/vae/data/'):
            os.mkdir('experimentos/vae/data/')
        if not os.path.isdir('experimentos/vae/metrics/'):
            os.mkdir('experimentos/vae/metrics/')
        if not os.path.isdir('experimentos/vae/models/'):
            os.mkdir('experimentos/vae/models/')
        if not os.path.isdir('experimentos/vae/plots/'):
            os.mkdir('experimentos/vae/plots/')
        path='experimentos/vae/'
    elif net =="dcvae":
        if not os.path.isdir('experimentos/dcvae/'):
            os.mkdir('experimentos/dcvae/')
        if not os.path.isdir('experimentos/dcvae/data/'):
            os.mkdir('experimentos/dcvae/data/')
        if not os.path.isdir('experimentos/dcvae/metrics/'):
            os.mkdir('experimentos/dcvae/metrics/')
        if not os.path.isdir('experimentos/dcvae/models/'):
            os.mkdir('experimentos/dcvae/models/')
        if not os.path.isdir('experimentos/dcvae/plots/'):
            os.mkdir('experimentos/dcvae/plots/')
        path='experimentos/dcvae/'
    elif net =="cvae":
        if not os.path.isdir('experimentos/cvae/'):
            os.mkdir('experimentos/cvae/')
        if not os.path.isdir('experimentos/cvae/data/'):
            os.mkdir('experimentos/cvae/data/')
        if not os.path.isdir('experimentos/cvae/metrics/'):
            os.mkdir('experimentos/cvae/metrics/')
        if not os.path.isdir('experimentos/cvae/models/'):
            os.mkdir('experimentos/cvae/models/')
        if not os.path.isdir('experimentos/cvae/plots/'):
            os.mkdir('experimentos/cvae/plots/')
        path='experimentos/cvae/'
    elif net =="gan":
        if not os.path.isdir('experimentos/gan/'):
            os.mkdir('experimentos/gan/')
        if not os.path.isdir('experimentos/gan/data/'):
            os.mkdir('experimentos/gan/data/')
        if not os.path.isdir('experimentos/gan/metrics/'):
            os.mkdir('experimentos/gan/metrics/')
        if not os.path.isdir('experimentos/gan/models/'):
            os.mkdir('experimentos/gan/models/')
        if not os.path.isdir('experimentos/gan/plots/'):
            os.mkdir('experimentos/gan/plots/')
        path='experimentos/gan/'
    elif net =="dcgan":
        if not os.path.isdir('experimentos/dcgan/'):
            os.mkdir('experimentos/dcgan/')
        if not os.path.isdir('experimentos/dcgan/data/'):
            os.mkdir('experimentos/dcgan/data/')
        if not os.path.isdir('experimentos/dcgan/metrics/'):
            os.mkdir('experimentos/dcgan/metrics/')
        if not os.path.isdir('experimentos/dcgan/models/'):
            os.mkdir('experimentos/dcgan/models/')
        if not os.path.isdir('experimentos/dcgan/plots/'):
            os.mkdir('experimentos/dcgan/plots/')
        path='experimentos/dcgan/'
    elif net =="cgan":
        if not os.path.isdir('experimentos/cgan/'):
            os.mkdir('experimentos/cgan/')
        if not os.path.isdir('experimentos/cgan/data/'):
            os.mkdir('experimentos/cgan/data/')
        if not os.path.isdir('experimentos/cgan/metrics/'):
            os.mkdir('experimentos/cgan/metrics/')
        if not os.path.isdir('experimentos/cgan/models/'):
            os.mkdir('experimentos/cgan/models/')
        if not os.path.isdir('experimentos/cgan/plots/'):
            os.mkdir('experimentos/cgan/plots/')
        path='experimentos/cgan/'
    elif net =="cgan_lineal":
        if not os.path.isdir('experimentos/cgan_lineal/'):
            os.mkdir('experimentos/cgan_lineal/')
        if not os.path.isdir('experimentos/cgan_lineal/data/'):
            os.mkdir('experimentos/cgan_lineal/data/')
        if not os.path.isdir('experimentos/cgan_lineal/metrics/'):
            os.mkdir('experimentos/cgan_lineal/metrics/')
        if not os.path.isdir('experimentos/cgan_lineal/models/'):
            os.mkdir('experimentos/cgan_lineal/models/')
        if not os.path.isdir('experimentos/cgan_lineal/plots/'):
            os.mkdir('experimentos/cgan_lineal/plots/')
        path='experimentos/cgan_lineal/'
    elif net =="wcgan":
        if not os.path.isdir('experimentos/wcgan/'):
            os.mkdir('experimentos/wcgan/')
        if not os.path.isdir('experimentos/wcgan/data/'):
            os.mkdir('experimentos/wcgan/data/')
        if not os.path.isdir('experimentos/wcgan/metrics/'):
            os.mkdir('experimentos/wcgan/metrics/')
        if not os.path.isdir('experimentos/wcgan/models/'):
            os.mkdir('experimentos/wcgan/models/')
        if not os.path.isdir('experimentos/wcgan/plots/'):
            os.mkdir('experimentos/wcgan/plots/')
        path='experimentos/wcgan/'
    elif net =="seq2seq":
        if not os.path.isdir('experimentos/seq2seq/'):
            os.mkdir('experimentos/seq2seq/')
        if not os.path.isdir('experimentos/seq2seq/data/'):
            os.mkdir('experimentos/seq2seq/data/')
        if not os.path.isdir('experimentos/seq2seq/metrics/'):
            os.mkdir('experimentos/seq2seq/metrics/')
        if not os.path.isdir('experimentos/seq2seq/models/'):
            os.mkdir('experimentos/seq2seq/models/')
        if not os.path.isdir('experimentos/seq2seq/plots/'):
            os.mkdir('experimentos/seq2seq/plots/')
        path='experimentos/seq2seq/'
    elif net =="cond_s2s":
        if not os.path.isdir('experimentos/cond_s2s/'):
            os.mkdir('experimentos/cond_s2s/')
        if not os.path.isdir('experimentos/cond_s2s/data/'):
            os.mkdir('experimentos/cond_s2s/data/')
        if not os.path.isdir('experimentos/cond_s2s/metrics/'):
            os.mkdir('experimentos/cond_s2s/metrics/')
        if not os.path.isdir('experimentos/cond_s2s/models/'):
            os.mkdir('experimentos/cond_s2s/models/')
        if not os.path.isdir('experimentos/cond_s2s/plots/'):
            os.mkdir('experimentos/cond_s2s/plots/')
        path='experimentos/cond_s2s/'
    elif net =="cond_s2s_conv":
        if not os.path.isdir('experimentos/cond_s2s_conv/'):
            os.mkdir('experimentos/cond_s2s_conv/')
        if not os.path.isdir('experimentos/cond_s2s_conv/data/'):
            os.mkdir('experimentos/cond_s2s_conv/data/')
        if not os.path.isdir('experimentos/cond_s2s_conv/metrics/'):
            os.mkdir('experimentos/cond_s2s_conv/metrics/')
        if not os.path.isdir('experimentos/cond_s2s_conv/models/'):
            os.mkdir('experimentos/cond_s2s_conv/models/')
        if not os.path.isdir('experimentos/cond_s2s_conv/plots/'):
            os.mkdir('experimentos/cond_s2s_conv/plots/')
        path='experimentos/cond_s2s_conv/'
    elif net =="dcnn":
        if not os.path.isdir('experimentos/dcnn/'):
            os.mkdir('experimentos/dcnn/')
        if not os.path.isdir('experimentos/dcnn/data/'):
            os.mkdir('experimentos/dcnn/data/')
        if not os.path.isdir('experimentos/dcnn/metrics/'):
            os.mkdir('experimentos/dcnn/metrics/')
        if not os.path.isdir('experimentos/dcnn/models/'):
            os.mkdir('experimentos/dcnn/models/')
        if not os.path.isdir('experimentos/dcnn/plots/'):
            os.mkdir('experimentos/dcnn/plots/')
        path='experimentos/dcnn/'
    else:
        sys.exit("You must choose a valid model...")
    return path

def moving_files(loss_function):
    "Creando nuevo folder y moviendo los resultados generados"
    import shutil
    import os 
    new_folder = "experimentos/"+loss_function
    if not os.path.isdir(new_folder): #creando new_folder
      os.mkdir(new_folder) 

      dest = new_folder
      print("Moving files to :", dest)
      shutil.move('experimentos/data', dest)
      shutil.move('experimentos/plots', dest)
      shutil.move('experimentos/models', dest) 
      shutil.move('experimentos/metrics', dest)

def dwprobability(test_true,test_pred,seq_length,num_features,path,net):
    """Función para calcular y graficar la dimension_wise_probability"""

    test_true = np.array(test_true).reshape(-1,num_features,seq_length)
    test_pred = np.array(test_pred).reshape(-1,num_features,seq_length)

    prob_real = np.mean(test_true, axis=0)
    prob_syn = np.mean(test_pred, axis=0)

    #plotting
    p1 = plt.scatter(prob_real, prob_syn, c="b", alpha=0.5, label=str(net))
    x_max = max(np.max(prob_real), np.max(prob_syn))
    x = np.linspace(0, x_max + 0.5)
    p2 = plt.plot(x, x, linestyle='--', color='gray', label="Ideal")  # solid
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.tick_params(labelsize=10)
    plt.legend(loc=2, prop={'size': 10})
    plt.title('Dimension-wise probability performance')
    plt.xlabel('Real data')
    plt.ylabel('Synth data')
    plt.savefig(path +'plots/'+"dim_wise_proba.png")
    plt.close()

def rescale_data(np_data,seq_length,num_features,path):
    #loading scaler
    scaler=pickle.load(open(path+'data/'+'scaler','rb'))

    np_array_rescale= scaler.inverse_transform(np.reshape(np_data,(-1,seq_length*num_features)))
    np_array_rescale = np.around(np_array_rescale,2)
    np_array_rescale = np_array_rescale.reshape(-1,seq_length,num_features)
    return np_array_rescale

def plotting_predictions(real_input,real_target,pred,seq_length,
                         num_features,feature_list,path, net, conditions,
                         num_conditions):
    real_input = np.array(real_input).reshape(-1,seq_length,num_features)
    real_target = np.array(real_target).reshape(-1,seq_length,num_features)
    pred = np.array(pred).reshape(-1,seq_length,num_features)

    past_rescale = rescale_data(real_input,seq_length,num_features,path)#[32, 24, 8])
    expected_rescale = rescale_data(real_target,seq_length,num_features,path)#[32, 24, 8])
    pred_rescale = rescale_data(pred,seq_length,num_features,path)#[32, 24, 8])

    # #selección de 5 índices aleatorios from pred_seqs
    indices = np.random.choice(range(real_input.shape[0]),replace=False,size=5)

    for index in indices: #recorriendo los indices
        past_pat = past_rescale[index, :, :] #(24, 8)
        expected_pat = expected_rescale[index,:, :] #(24, 8)
        prediction_pat= pred_rescale[index,:,:] #(24, 8)
        #print("prediction_pat:",prediction_pat)

        if net == "cgan" or net == "cond_s2s" or net == "cvae" or net =="cond_s2s_conv" or net=="cgan_lineal":
            np_cond = np.array(conditions).reshape(-1,num_conditions)
            conditions_per_patient = np_cond[index]
            age = conditions_per_patient[0]
            sex = conditions_per_patient[1].astype(int)

            if num_conditions>2:
                dm = conditions_per_patient[2].astype(int)
                hyp = conditions_per_patient[3].astype(int)
                hd = conditions_per_patient[4].astype(int)        

        for index_feature in range(len(feature_list)):
            past = past_pat[:,index_feature]
            expected = expected_pat[:,index_feature]
            prediction = prediction_pat[:,index_feature]
            
            plt.rcParams['figure.figsize'] = (17.0,5.0)
            fig, ax = plt.subplots()
            ax.plot(range(len(past)),past, "r.-", label="History", color = "blue")

            ax.plot(range(len(past)-1, len(expected) + len(past)),np.concatenate([past[len(past)-1:len(past)], expected]),
                    "r.-",label="Ground truth",color = "orange")
            
            ax.plot(range(len(past)-1, len(prediction) + len(past)),np.concatenate([past[len(past)-1:len(past)], prediction]), 
                    "r.-", label="Prediction", color="green")
            ax.set_xlabel("Hours")
            ax.set_ylabel("Values")
            
            if net == "cgan" or net=="cond_s2s" or net == "cvae" or net =="cond_s2s_conv" or net=="cgan_lineal":
                if sex.astype(int) == 0:
                    sex_value = "Woman"
                else:
                    sex_value = "Man"
                if num_conditions > 2:
                    ax.legend(title='Sex: '+str(sex_value)+' Age:'+str(age)+ '\n'
                          +' DM:'+str(dm) +' HYP:'+str(hyp)+' HeartD:'+str(hd))
                else:
                    ax.legend(title='Sex: '+str(sex_value)+' Age:'+str(age))
            else:
                ax.legend()
            
            ax.set_title("Timeseries forecasting "+"("+str(feature_list[index_feature])+")")
            plt.savefig(path+'plots/'+"plot_"+str(index)+"_"+str(feature_list[index_feature])+".png")
            plt.close()

def extracting_demo_data_from_scratch(df,num_features,window_size,path,filename):
    "Steps for extracting (from scrath) demo_data"

    path_data = "/home/blanca/anaconda3/envs/synthm/RGAN/experiments_seq2seq/database/"
    demo_data = pd.read_csv(path_data+"mimic_demo_diagnoses.csv")

    patients = set(df.pid)
    pd_demo=pd.DataFrame(columns=['age','gender'])
 
    for pat in patients:
        age,gender=get_demo_data(pat)
        values = {"pid":pat,"age":age.item(),"gender":gender.item()}
        pd_demo = pd_demo.append(values,ignore_index=True) #saving demo data

    pd_demo.to_csv(path+str(filename)+"_demo_data.csv")


def loading_data_for_conditional(seq_length,feature_list,path):
    train,val = get_df("iterative")

    #extracting pids + offset
    train_pids = train[['pid','offset']]
    val_pids = val[['pid','offset']]

    train_data = train[feature_list]
    val_data = val[feature_list]

    #Scaling data
    Xtrain = np.asarray(train_data).reshape(-1,seq_length*train_data.shape[1])
    Xval = np.asarray(val_data).reshape(-1,seq_length*val_data.shape[1])

    scale_range=(0, 1)
    scaler = MinMaxScaler(feature_range=scale_range).fit(np.vstack([Xtrain]))

    scaled_train = scaler.transform(Xtrain).reshape(-1*train_data.shape[0],train_data.shape[1])
    scaled_val = scaler.transform(Xval).reshape(-1* val_data.shape[0],val_data.shape[1])

    pd_train = pd.DataFrame(scaled_train,columns = train_data.columns)
    pd_val = pd.DataFrame(scaled_val,columns = val_data.columns)

    #concat scaled data + pids
    pd_train = pd.concat([train_pids, pd_train], axis=1)
    pd_val = pd.concat([val_pids, pd_val], axis=1)

    #Saving scaler
    pickle.dump(scaler, open(path+'data/'+'scaler', 'wb'))
    
    return pd_train,pd_val