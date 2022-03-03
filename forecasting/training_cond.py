#source: https://github.com/CVxTz/time_series_forecasting
import os
import json
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

#Loading models
from models.cgan_lineal_conv import CGAN_lineal
from models.cgan import CGAN
from models.wcgan import WCGAN
from models.cond_s2s import CondS2S,encoder, decoder
from models.cvae import COND_dcvae
from models.cond_s2s_conv import CondS2S_conv, encoder, decoder

#Loading losses & utilities
from loss.losses import r2_corr, rmse_loss
from loss.dilate_loss import dilate_loss
from utils import create_folders, dwprobability, plotting_predictions, \
                  loading_data_for_conditional

import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

start_time = datetime.datetime.now()
print("Start time:",start_time)

def split_df(df: pd.DataFrame,history_size: int = 24):
    """
    Create a training / validation samples
    Validation samples are the last seq_length rows
    """
    seq_length=history_size #24
    end_index = df.shape[0]

    #print("end_index:", end_index, "seq_length:",seq_length) #Si, end_index = 48
    label_index = end_index - seq_length    #then: 48-24 = 24
    start_index = max(0, label_index - history_size) #then: (0,24-24) = 0

    #verificado: si devuelve las primeras (history) y las siguientes (targets) horas
    history = df[start_index:label_index] #0-24 (first 24 hours)
    targets = df[label_index:end_index] #(24-48) #next 24 hours

    return history, targets


def pad_arr(arr: np.ndarray, expected_size: int = 24):
    """
    Pad top of array when there is not enough history
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode='edge')
    return arr


def df_to_np(df):
    """
    Convert dataframe to numpy
    """
    arr = np.array(df)
    arr = pad_arr(arr)
    return arr


class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, target,demo):
        self.groups = groups
        self.grp_by = grp_by
        self.features = target
        self.demo = demo

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        """
        targets_in = first 24 hours
        targets_out = next 24 hours
        """
        list_targets_in=[]
        pid_patient = self.groups[idx]

        df = self.grp_by.get_group(pid_patient) #get all features for each patient
        df_conditions = self.demo.get_group(pid_patient)
        df_conditions = df_conditions[['age','gender','diabetes','hypertension','heart_diseases']]

        #history=first24hrs, targets=next24hours
        history, targets = split_df(df) #history (24,10), targets (24,10) 

        targets_in = history[self.features] #(24,8), first24hours
        targets_in = df_to_np(targets_in)#(24,8)
        targets_in = targets_in.reshape(24,8)

        targets_out = targets[self.features] #(24,8)#next24hours
        targets_out = np.array(targets_out) #(24,8)
        targets_out = targets_out.reshape(24,8)

        conditions = np.around(np.array(df_conditions),1) #sex, age
        conditions = conditions.flatten()

        targets_in = torch.tensor(targets_in, dtype=torch.float) #torch.Size([24, 8])
        targets_out = torch.tensor(targets_out, dtype=torch.float) #torch.Size([24, 8])
        conditions = torch.tensor(conditions,dtype=torch.float) #[1,2]

        return targets_in, targets_out,conditions


def train(data_train,data_val,
          database: str,
          feature_list: str,
          net:str,
          output_json_path: str,
          conditions: str,
          log_dir: str = 'ts_logs',
          model_dir: str = 'ts_models',
          batch_size: int = 10,
          epochs: int = 10,
          learning_rate: float = 1e-3,
          decay: float = 1e-3,
          alpha: float = 1e-3,
          gamma: float = 1e-3,
          seq_length: int = 24,
          num_features: int=8,):
    
    feature_target_names_path = "data/"
    print("Epochs:",epochs,"Batch_size:",batch_size, "conditions:", conditions)
    target_vars = feature_list

    if net == "cond_s2s" or net == "cond_s2s_conv": 
        train_np = data_train.to_numpy().reshape(-1,(seq_length*2)*data_train.shape[1]) #23520
        train_np = train_np[:23520].reshape(-1*(seq_length*2),data_train.shape[1])

        data_train = pd.DataFrame(train_np, columns = data_train.columns)

        val_np = data_val.to_numpy().reshape(-1,(seq_length*2)*data_val.shape[1]) #23520
        val_np = val_np[:7840].reshape(-1*(seq_length*2),data_val.shape[1])

        data_val = pd.DataFrame(val_np, columns = data_val.columns)

    #------------Opening demo data --------------------#
    demo = pd.read_csv("/home/blanca/anaconda3/envs/synthm/RGAN/experiments_seq2seq/database/mimic_demo_diagnoses.csv")

    #join: Cardiac dysrhythmias + Cardiac arrest and ventricular fibrillation
    demo['heart_diseases'] = demo['Cardiac dysrhythmias'] + demo['Cardiac arrest and ventricular fibrillation']
    #Selection of demographic data for conditioning
    demo['age'] = round(demo.AGE,1)
    demo['gender'] = np.where(demo['GENDER'] == 'M', 1, 0).astype(int)
    
    demo['hypertension'] = round(demo['Hypertension'],1)
    demo['diabetes'] = round(demo['Diabetes mellitus'],1)
    demo['heart_diseases'] = np.where(demo['heart_diseases']>= 1, 1, 0).astype(int)
    
    train_demo = demo[['PID','age','gender','diabetes','hypertension','heart_diseases']]
    val_demo = demo[['PID','age','gender','diabetes','hypertension','heart_diseases']]
    #---------------------------------------------------#

    #data
    grp_by_train = data_train.groupby(by=['pid'])
    grp_by_val = data_val.groupby(by=['pid'])
    
    #demographic data
    grp_by_train_demo = train_demo.groupby(by=['PID'])
    grp_by_val_demo = val_demo.groupby(by=['PID'])

    groups_train = list(grp_by_train.groups)
    groups_val= list(grp_by_val.groups)
    groups_train_demo = list(grp_by_train_demo.groups)
    groups_val_demo= list(grp_by_val_demo.groups)

    full_groups_train = [grp for grp in groups_train if grp_by_train.get_group(grp).shape[0]>=2*seq_length]
    full_groups_val = [grp for grp in groups_val if grp_by_val.get_group(grp).shape[0]>=2*seq_length]

    train_data = Dataset(groups=full_groups_train,
                         grp_by=grp_by_train,
                         target=target_vars,#son todas las vars, menos pids y offset
                         demo = grp_by_train_demo) #sex,gender

    val_data = Dataset(groups=full_groups_val,
                       grp_by=grp_by_val,
                       target=target_vars,#son todas las vars, menos pids y offset
                       demo = grp_by_val_demo)#sex,gender

    print("Train size:", len(train_data),"Val size:", len(val_data))


    #Dataloader  = #num_samples/batch_size
    train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=6,shuffle=True) 
    val_loader = DataLoader(val_data,batch_size=batch_size,num_workers=5,shuffle=False)
    print("Train (loader):",len(train_loader),"Val (loader):",len(val_loader)) 
    
    if net == "cgan_lineal":
        model = CGAN_lineal(w_decay=decay,alpha=alpha,gamma=gamma,
                    target_length=seq_length, num_features=num_features,
                    lr = learning_rate).to(device)

    elif net == "cgan": #el G inicia con una Conv1D
        model = CGAN(w_decay=decay,alpha=alpha,gamma=gamma,
                    target_length=seq_length, num_features=num_features,
                    num_conditions=train_demo.shape[1]-1,
                    lr = learning_rate).to(device)

    elif net == "cond_s2s": 
        enc = encoder(input_size=8, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size)
        dec = decoder(input_size=8, hidden_size=128, num_grulstm_layers=1, fc_units=16, output_size=8)
        model = CondS2S(encoder = enc,decoder = dec,w_decay=decay,dropout=0.1,
                        alpha=alpha,gamma=gamma,target_length=seq_length, num_features=num_features,
                        lr = learning_rate).to(device)

    elif net == "cond_s2s_conv":
        enc = encoder(input_size=8, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size)
        dec = decoder(input_size=8, hidden_size=128, num_grulstm_layers=1, fc_units=16, output_size=8)
        model = CondS2S_conv(encoder = enc,decoder = dec,w_decay=decay,dropout=0.1,
                        alpha=alpha,gamma=gamma,target_length=seq_length, num_features=num_features,
                        lr = learning_rate).to(device)

    elif net == "cvae": 
        model = COND_dcvae(w_decay=decay,dropout=0.1,
                alpha=alpha,gamma=gamma,target_length=seq_length, 
                num_features=num_features,lr = learning_rate).to(device)

    elif net == "wcgan": 
        model = WCGAN(w_decay=decay,alpha=alpha,gamma=gamma,
                    target_length=seq_length, num_features=num_features,
                    lr = learning_rate).to(device)
    else:
        sys.exit("You must choose a valid model...") 
    return model, train_loader, val_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--output_json_path", default=None)
    parser.add_argument("--log_dir")
    parser.add_argument("--model_dir")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decay",type=float,default=1e-3)
    parser.add_argument("--database")
    parser.add_argument("--net", type=str)
    parser.add_argument("--alpha",type=float,default=1e-2)
    parser.add_argument("--gamma",type=float,default=1e-2)
    parser.add_argument("--seq_length",type=int,default=24)
    parser.add_argument("--num_features",type=int,default=8)
    parser.add_argument("--conditions",default="age")

    args = parser.parse_args()
    path = create_folders(args.net) #creating folders to save results
    print("*** Saving results in: ", path, "***")

    #list for saving results
    list_rmse, list_dilate, list_shape, list_temporal=[],[],[],[]
    list_past,list_pred, list_real,list_r2, list_cond = [],[],[],[],[]
    seed_everything(42) #for reproducibility

    print("---------------------Starting:::",args.net,"-----------------")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    feature_list = ['arterial_bp_mean','respiratory_rate', 'diastolic_bp',
                    'spo2', 'o2_saturation','heart_rate', 'systolic_bp', 
                    'temperature']

    data_train, data_val = loading_data_for_conditional(args.seq_length,feature_list,path)

    model, train_loader, val_loader = train(data_train,data_val,
                                            database=args.database,
                                            feature_list = feature_list,
                                            net = args.net,
                                            output_json_path=args.output_json_path,
                                            log_dir=args.log_dir,
                                            model_dir=args.model_dir,
                                            epochs=args.epochs,
                                            batch_size=args.batch_size,
                                            learning_rate=args.learning_rate,
                                            decay=args.decay,
                                            alpha=args.alpha,
                                            gamma=args.gamma,
                                            seq_length=args.seq_length,
                                            num_features=args.num_features,
                                            conditions = args.conditions)

    es_callback = EarlyStopping(patience=10, verbose=1, min_delta=0.001, monitor='loss_val', mode='min')

    checkpoint_callback = ModelCheckpoint(monitor='loss_val',mode="min",
                                          dirpath=path+args.model_dir,
                                          filename=str(args.net)+'_{epoch:02d}-{val_loss:.2f}',)

    logger = TensorBoardLogger(save_dir=path+args.log_dir, version=1, name="logs")

    trainer = Trainer(max_epochs=args.epochs,
                      gpus=1 if torch.cuda.is_available() else 0,
                      logger = False,
                      callbacks=[es_callback,checkpoint_callback],
                      progress_bar_refresh_rate=50,
                      gradient_clip_val=0.1)

    print("------------------ Training --------------------------")
    trainer.fit(model, train_loader, val_loader)
    
    print("------------------ Hparams ---------------------------")
    print(model.hparams)

    print("----------------- Predictions ------------------------")    
    model.eval()
    with torch.no_grad(): #turn off gradients computation

        for i,data in enumerate(val_loader):
            
            #(first 24hrs, next 24hrs, conditions)
            real_input, real_target, conditions = data #conditions: torch.Size([32, 2])
            
            past = torch.tensor(real_input, dtype=torch.float32).to(device).reshape(-1,args.seq_length,args.num_features) #[32, 24,8])
            
            expected = torch.tensor(real_target, dtype=torch.float32).to(device).reshape(-1,args.seq_length,args.num_features) #[32, 24,8])
            
            conditions = torch.tensor(conditions, dtype=torch.float32).to(device)
            
            model = model.to(device)

            if args.net == "cvae":
                pred,_,_ = model(past,conditions) #[32, 24, 8])
            else:
                ##Only for testing: si past es un tensor de números aletorios, ¿la pred es la misma?
                #past = torch.rand((32,24,8)).to(device)
                pred = model(past,conditions) # y_pred [32,24,8]

            rmse = rmse_loss(pred,expected)

            loss,loss_shape,loss_temporal = dilate_loss(pred,expected,alpha=args.alpha,
                                                        gamma=args.gamma,device=device)

            score_R2 = r2_corr(pred.view(-1).detach().cpu().numpy().reshape(-1,1),
                               expected.view(-1).detach().cpu().numpy().reshape(-1,1))

            list_rmse.append(rmse.item())
            list_dilate.append(loss.item())
            list_shape.append(loss_shape.item())
            list_temporal.append(loss_temporal.item())
            list_r2.append(score_R2)

            #extend
            list_past.extend(past.view(-1).detach().cpu().numpy().reshape(-1,1))
            list_real.extend(expected.view(-1).detach().cpu().numpy().reshape(-1,1))
            list_pred.extend(pred.view(-1).detach().cpu().numpy().reshape(-1,1))
            list_cond.extend(conditions.view(-1).detach().cpu().numpy().reshape(-1,1))

    print("RMSE:", round(np.mean(list_rmse),3))
    print("Dilate:", round(np.mean(list_dilate),3))
    print("Shape:", round(np.mean(list_shape),3))
    print("Temporal:", round(np.mean(list_temporal),3))
    print("R2:", round(np.mean(list_r2),3))

    dwprobability(list_real,list_pred,args.seq_length,args.num_features,path,args.net)

    plotting_predictions(list_past,list_real,list_pred,args.seq_length,args.num_features,feature_list,path,
                         args.net,list_cond,conditions.shape[1])

    #---- Saving results ----#
    output_json = { 
                    "loss_rmse":round(np.mean(list_rmse),3),
                    "loss_dilate":round(np.mean(list_dilate),3),
                    "loss_shape":round(np.mean(list_shape),3),
                    "loss_temporal":round(np.mean(list_temporal),3),
                    "R2":round(np.mean(list_r2),3),
                    "best_model_path": checkpoint_callback.best_model_path,
                    "learning_rate":model.hparams.lr,
                    "alpha":model.hparams.alpha,
                    "gamma":model.hparams.gamma}

    output_json_path = path + args.output_json_path+"_"+str(args.net)+".json"
    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

print("End time:", datetime.datetime.now() - start_time)

