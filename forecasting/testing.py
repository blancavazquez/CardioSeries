"""
Code for testing the model with STEMI & NSTEMI patients
Author: Blanca Vázquez
"""

import os
import re
import json
import pickle
import random
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Loading pytorch_lightning
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

#Loading losses & utilities
from loss.losses import r2_corr, rmse_loss
from loss.dilate_loss import dilate_loss
from torchmetrics import R2Score
from utils import create_folders, dwprobability_con_metricas, plotting_predictions, \
                  loading_eicu,plotting_losses,loading_data_for_testing, \
                  computing_rmse,saving_metrics, compute_metrics, \
                  extracting_demo_data_from_scratch, extracting_demo_mimic,\
                  splitting_data, xgb_model,saving_predictions, \
                  plotting_rmse,saving_files_for_mortality

#Loading models
from models.dcnn_causal import CausalModel

import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

start_time = datetime.datetime.now()
print("Start time:",start_time)

def split_df(df,in_seqlength,out_seqlength):
    """Split data: first hours (history), next hours (targets)"""
    start_index = 0
    end_index = in_seqlength+out_seqlength
    history = df[start_index:in_seqlength]
    targets = df[in_seqlength:end_index]
    return history, targets

def pad_arr(arr, expected_size):
    """
    Pad top of array when there is not enough history
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode='edge')
    return arr


def df_to_np(df,seq_length):
    """
    Convert dataframe to numpy
    """
    arr = np.array(df)
    arr = pad_arr(arr,seq_length)
    return arr

class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, features,demo, mask_saits,
                 input_seq_length,output_seq_length,num_features,database):
        self.groups = groups
        self.grp_by = grp_by
        self.features = features
        self.demo = demo
        self.mask_saits = mask_saits
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.num_features = num_features
        self.database = database

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        """ targets_in = first 24 hours, targets_out = next 24 hours """
        list_targets_in=[]
        pid_patient = self.groups[idx]

        df = self.grp_by.get_group(pid_patient) #get all features for each patient

        df_conditions = self.demo.get_group(pid_patient)

        conditions = pd.DataFrame()
    
        if self.database == 'mimic':
            df_conditions = df_conditions[['age','gender','diabetes','hypertension','heart_diseases',
                                           'tsicu','sicu','micu','ccu','csru']]
            conditions = np.around(np.array(df_conditions),1) #sex, age
            conditions = torch.tensor(conditions,dtype=torch.float) #[1,2]

        else: #eicu
            df_conditions = df_conditions[['age','gender','diabetes','hypertension','heart_diseases',
                                           'tsicu','sicu','micu','ccu','csru']]
            conditions = np.array(df_conditions)            
            conditions = torch.tensor(conditions,dtype=torch.float) #[1,2]

        #----- Get EHR real ------#
        #history=first24hrs, targets=next24hours
        history, targets = split_df(df,self.input_seq_length,self.output_seq_length)
        targets_in = history[self.features]
        targets_in = df_to_np(targets_in,self.input_seq_length)
        targets_in = targets_in.reshape(self.input_seq_length,self.num_features)
        targets_past = torch.tensor(targets_in, dtype=torch.float)

        targets_out = targets[self.features]
        targets_out = np.array(targets_out)
        targets_out = targets_out.reshape(self.output_seq_length,self.num_features)
        targets_expected = torch.tensor(targets_out, dtype=torch.float)

        df_mask = self.mask_saits.get_group(pid_patient)
        df_mask = df_mask[self.features]

        #"---Masking only for next 24 hours----"# 
        history_mask, targets_mask = split_df(df_mask,self.input_seq_length,self.output_seq_length)
        targets_mask = targets_mask.to_numpy()
        mask = torch.tensor(targets_mask,dtype=torch.float).eq(1).reshape(targets_mask.shape[0],targets_mask.shape[1]) #[6,8]
        conditions = conditions.flatten()

        return targets_past, targets_expected,conditions, mask


def test(data_val,val_mask,
          database: str,
          feature_list: str,
          net:str,
          conditions: str,
          log_dir: str = 'ts_logs',
          model_dir: str = 'ts_models',
          batch_size: int = 10,
          learning_rate: float = 1e-3,
          decay: float = 1e-3,
          alpha: float = 1e-3,
          gamma: float = 1e-3,
          input_seq_length: int = 6,
          output_seq_length: int = 6,
          num_features: int=8, 
          path: str = 'data',
          missing_values: str = 'False',
          saits_impute: str = 'False'):
    
    feature_target_names_path = "data/"
    print("Batch_size:",batch_size, "conditions:", conditions)
    seq_length = 6 
    val_np = data_val.to_numpy().reshape(-1,(seq_length*2)*data_val.shape[1])
    val_np = val_np.reshape(-1*(seq_length*2),data_val.shape[1])
    data_val = pd.DataFrame(val_np, columns = data_val.columns)

    #----------- Selection of demographic data for conditioning --------------------#
    if database == 'mimic':
        demo_data, data_val = extracting_demo_mimic(data_val,feature_list)
        print("demo_data:", demo_data.shape, "EHR:", data_val.shape)
        target_vars = feature_list        
    else: #eicu
        demo_data, data_val = extracting_demo_data_from_scratch(data_val)
        print("demo_data:", demo_data.shape,"EHR:", data_val.shape)
        target_vars = feature_list

    #---------------------------------------------------#
    grp_by_val = data_val.groupby(by=['pid'])
    groups_val= list(grp_by_val.groups)
        
    #demographic data
    grp_by_demo = demo_data.groupby(by=['PID'])

    if saits_impute == "True":
        grp_by_val_mask = val_mask.groupby(by=['pid'])
    else:
        grp_by_val_mask=[],[]

    full_groups_val = [grp for grp in groups_val if grp_by_val.get_group(grp).shape[0]>=2*seq_length]

    val_data = Dataset(groups=full_groups_val,grp_by=grp_by_val,
                       features=target_vars,
                       demo = grp_by_demo,
                       mask_saits = grp_by_val_mask,
                       input_seq_length = input_seq_length,
                       output_seq_length = output_seq_length,
                       num_features = num_features,
                       database = database)
    val_loader = DataLoader(val_data,batch_size=batch_size,num_workers=5,shuffle=False)
    print("Val size:", len(val_data),"Val (loader):",len(val_loader))
    return val_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--foldername")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decay",type=float,default=1e-3)
    parser.add_argument("--net", type=str)
    parser.add_argument("--database", type=str)
    parser.add_argument("--alpha",type=float,default=1e-2)
    parser.add_argument("--gamma",type=float,default=1e-2)
    parser.add_argument("--input_seq_length",type=int,default=6)
    parser.add_argument("--output_seq_length",type=int,default=6)
    parser.add_argument("--output_rnn",type=int,default=7)
    parser.add_argument("--num_features",type=int,default=8)
    parser.add_argument("--conditions",type=int,default=2)
    parser.add_argument("--subpopulation",type=str,default="stemi")
    parser.add_argument("--missing_values",default="False")
    parser.add_argument("--saits_impute",default="False")
    parser.add_argument("--model_name",default="False")
    parser.add_argument("--feature_list",default="temperature")
    args = parser.parse_args()
    feature_list = list(args.feature_list.split(" "))

    print("*************************************************************")
    print("*************************************************************")
    path = create_folders(args.foldername, args.input_seq_length, args.output_seq_length,args.conditions)
    print("*** Saving results in: ", path, "***")

    #list for saving results
    list_rmse, list_dilate, list_shape, list_temporal=[],[],[],[]
    list_past,list_pred, list_real,list_r2, list_cond = [],[],[],[],[]
    list_mask = []
    seed_everything(42) #for reproducibility

    print("---------------------Starting:::",args.net,"-----------------")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    features = ['arterial_bp_mean','respiratory_rate', 'diastolic_bp','spo2','heart_rate', 'systolic_bp', 'temperature']

    model = CausalModel(w_decay=args.decay,dropout=0.1,alpha=args.alpha,gamma=args.gamma,input_seq_length=args.input_seq_length,
                        output_seq_length=args.output_seq_length,output_rnn=args.output_rnn,num_features=args.num_features,batch_size=args.batch_size,lr = args.learning_rate,
                        num_conditions=args.conditions,path = path, feature_list= feature_list,net = args.net,
                        in_channels=32,out_channels=2,kernel_size=1,stride=1,dilation=1,groups=1,bias=False).to(device)

    if args.database == 'mimic':
        test_data,test_mask,features = loading_data_for_testing(args.input_seq_length,features,path,
                                                                args.missing_values, args.saits_impute,
                                                                args.subpopulation)
    else:
        test_data, test_mask,features = loading_eicu(args.input_seq_length,features,path,
                                                    args.missing_values, args.saits_impute,
                                                    args.subpopulation)
    print("test_data:",test_data.shape,"test_mask:",test_mask.shape,"features:",len(features))
    val_loader = test(test_data,test_mask,database=args.database,
                      feature_list = features,net = args.net,
                      batch_size=args.batch_size,
                      learning_rate=args.learning_rate, decay=args.decay,
                      alpha=args.alpha,gamma=args.gamma,
                      input_seq_length=args.input_seq_length,
                      output_seq_length=args.output_seq_length,
                      num_features=args.num_features,
                      conditions = args.conditions,path = path,
                      missing_values = args.missing_values,
                      saits_impute = args.saits_impute)

    print("------------------ Loading model --------------------------")
    path_model = path+"models/"+args.model_name+".ckpt"
    print("Loading model from:", path_model)
    
    model = model.load_from_checkpoint(path_model)

    print("--------------------- Hparams -----------------------------")
    print(model.hparams)

    print("--------------------- Model -----------------------------")
    print(model)

    list_rmse, list_dilate, list_shape, list_temporal=[],[],[],[]
    list_past,list_pred, list_real,list_r2, list_cond,list_mask = [],[],[],[],[],[]
    r2score = R2Score().to(device)

    print("-------------------- Predictions --------------------------")
    model.eval()    
    with torch.no_grad(): #turn off gradients computation

        for i,data in enumerate(val_loader): 
            target_in, target_out,condition, mask = data
            condition = condition[:,:args.conditions]
            past = torch.tensor(target_in, dtype=torch.float32).to(device)
            expected = torch.tensor(target_out, dtype=torch.float32).to(device)
            conditions = condition.to(device)
            mask = mask.to(device)
            model = model.to(device)

            # - Prediction - #
            pred = model(past,conditions)
            print("pred:", pred.shape)
            pred_mask = torch.masked_select(pred, mask)
            expected_mask = torch.masked_select(expected, mask)
            rmse = rmse_loss(pred_mask,expected_mask)
            r2_torch = r2score(pred_mask.reshape(-1).to(device),expected_mask.reshape(-1).to(device))    
            dilate,loss_shape,loss_temporal = dilate_loss(pred_mask,expected_mask,batch_size=args.batch_size,
                                                          seq_length = args.output_seq_length,num_features=args.num_features,
                                                          alpha=args.alpha,gamma=args.gamma,device=device,mask = "True")
            list_rmse.append(rmse.item())
            list_dilate.append(dilate.item())
            list_shape.append(loss_shape.item())
            list_temporal.append(loss_temporal.item())
            list_r2.append(r2_torch.item())

            list_past.extend(past.view(-1).detach().cpu().numpy().reshape(-1,1))
            list_real.extend(expected.view(-1).detach().cpu().numpy().reshape(-1,1))
            list_pred.extend(pred.reshape(-1).detach().cpu().numpy().reshape(-1,1))
            list_cond.extend(conditions.reshape(-1).detach().cpu().numpy().reshape(-1,1))
                list_mask.extend(mask.view(-1).detach().cpu().numpy().reshape(-1,1))

    saving_metrics(list_rmse,list_dilate,list_shape,list_temporal,list_r2,path) 
    computing_rmse(list_real,list_pred,args.output_seq_length,features,"escalado",path)
    computing_rmse(list_real,list_pred,args.output_seq_length,features,"sin_escalar",path)
    dwprobability_con_metricas(list_real,list_pred,list_rmse,list_r2,args.output_seq_length,args.num_features,path,args.net,"testing",1)
    plotting_predictions(list_past,list_real,list_pred,list_mask,args.input_seq_length,args.output_seq_length,args.num_features,
                         args.conditions,list_cond,path,"dcnn",num_samples=pred.shape[0],mode="test",fold=0)
    plotting_rmse(list_real,list_pred,args.output_seq_length,features,"escalado",path)
    saving_files_for_mortality(list_real,list_pred,features,path)

print("End time:", datetime.datetime.now() - start_time, "Go to:",args.foldername)

