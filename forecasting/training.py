"""
Code for training model after hyperparameter selection
"""
import os
import datetime
import numpy as np
import pandas as pd

#Loading pytorch_lightning
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

#Loading models
from models.dcnn_causal import CausalModel

#Loading losses & utilities
from loss.losses import r2_corr, rmse_loss
from loss.dilate_loss import dilate_loss
from utils import create_folders, loading_data,plotting_losses,extraction_demo_data

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
    def __init__(self, groups, grp_by, target,demo, mask_saits,
                 input_seq_length, output_seq_length,num_features,path):
        self.groups = groups
        self.grp_by = grp_by
        self.features = target
        self.demo = demo
        self.mask_saits = mask_saits
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.num_features = num_features
        self.path = path

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        """ targets_in = first 24 hours, targets_out = next 24 hours """
        list_targets_in=[]
        pid_patient = self.groups[idx]

        df = self.grp_by.get_group(pid_patient) #get all features for each patient
        df_conditions = self.demo.get_group(pid_patient)
        df_conditions = df_conditions[['age','gender','diabetes','hypertension','heart_diseases']]

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
        history_mask, targets_mask = split_df(df_mask,self.input_seq_length,self.output_seq_length)
        targets_mask = targets_mask.to_numpy()
        mask = torch.tensor(targets_mask,dtype=torch.float).eq(1).reshape(targets_mask.shape[0],targets_mask.shape[1])

        # conditions
        conditions = np.around(np.array(df_conditions),1)
        conditions = torch.tensor(conditions,dtype=torch.float)
        conditions = conditions.flatten()        
        return targets_past, targets_expected,conditions, mask


def train(data_train,data_val,train_mask,val_mask,
          feature_list: str,
          net:str,
          conditions: str,
          batch_size: int = 10,
          epochs: int = 10,
          learning_rate: float = 1e-3,
          decay: float = 1e-3,
          alpha: float = 1e-3,
          gamma: float = 1e-3,
          input_seq_length: int=6,
          output_seq_length: int=6,
          output_rnn: int=7,
          num_features: int=7, 
          path: str = 'data',
          missing_values: str = 'False',
          saits_impute: str = 'False',
          device: str = 'cpu'):
    
    print("Epochs:",epochs,"Batch_size:",batch_size, "conditions:", conditions, "Out_seqlength:",output_seq_length)
    target_vars = list(feature_list.split(" "))
    seq_length = 6
    train_np = data_train.to_numpy().reshape(-1,(seq_length*2)*data_train.shape[1])
    train_np = train_np[:30208].reshape(-1*(seq_length*2),data_train.shape[1])
    data_train = pd.DataFrame(train_np, columns = data_train.columns)
    
    val_np = data_val.to_numpy().reshape(-1,(seq_length*2)*data_val.shape[1])
    val_np = val_np[:7680].reshape(-1*(seq_length*2),data_val.shape[1])
    data_val = pd.DataFrame(val_np, columns = data_val.columns)
    train_demo, val_demo = extraction_demo_data()
    
    print("Train:",data_train.shape,"Val:",data_val.shape)
    grp_by_train = data_train.groupby(by=['pid'])
    grp_by_val = data_val.groupby(by=['pid'])
    groups_train = list(grp_by_train.groups)
    groups_val= list(grp_by_val.groups)
    ##saits_impute == "True":
    grp_by_train_mask = train_mask.groupby(by=['pid'])
    grp_by_val_mask = val_mask.groupby(by=['pid'])
        
    #demographic data
    grp_by_train_demo = train_demo.groupby(by=['PID'])
    grp_by_val_demo = val_demo.groupby(by=['PID'])

    full_groups_train = [grp for grp in groups_train if grp_by_train.get_group(grp).shape[0]>=2*seq_length]
    full_groups_val = [grp for grp in groups_val if grp_by_val.get_group(grp).shape[0]>=2*seq_length]

    train_data = Dataset(groups=full_groups_train,grp_by=grp_by_train,
                        target=target_vars,
                        demo = grp_by_train_demo,
                        mask_saits = grp_by_train_mask,
                        input_seq_length = input_seq_length,
                        output_seq_length = output_seq_length,
                        num_features = num_features,path = path)

    val_data = Dataset(groups=full_groups_val,grp_by=grp_by_val,
                       target=target_vars,
                       demo = grp_by_val_demo,
                       mask_saits = grp_by_val_mask,
                       input_seq_length = input_seq_length,
                       output_seq_length = output_seq_length,
                       num_features = num_features,path = path)

    print("Train size (num patients):", len(train_data),"Val size (num patients):", len(val_data))
    train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=6,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,num_workers=5,shuffle=False)
    print("Train (loader):",len(train_loader),"Val (loader):",len(val_loader))
    
    model = CausalModel(w_decay=decay,dropout=0.1,alpha=alpha,gamma=gamma,input_seq_length=input_seq_length,
                        output_seq_length=output_seq_length,output_rnn = output_rnn, num_features=num_features,batch_size=batch_size,
                        lr = learning_rate,num_conditions=conditions,path = path, feature_list= target_vars,net = net,
                        in_channels=32,out_channels=2,kernel_size=1,stride=1,dilation=1,groups=1,bias=False).to(device)
    return model, train_loader, val_loader


if __name__ == "__main__":
    import argparse
    seed_everything(42) #for reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--foldername")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decay",type=float,default=1e-3)
    parser.add_argument("--net", type=str)
    parser.add_argument("--alpha",type=float,default=1e-2)
    parser.add_argument("--gamma",type=float,default=1e-2)
    parser.add_argument("--kfold",type=float,default=1e-2)
    parser.add_argument("--input_seq_length",type=int,default=12)
    parser.add_argument("--output_seq_length",type=int,default=12)
    parser.add_argument("--output_rnn",type=int,default=7)
    parser.add_argument("--num_features",type=int,default=8)
    parser.add_argument("--conditions",type=int,default=2)
    parser.add_argument("--missing_values",default="False")
    parser.add_argument("--saits_impute",default="False")
    parser.add_argument("--feature_list",type=str,default="temperature")
    args = parser.parse_args()
    feature_list = list(args.feature_list.split(" "))

    print("*************************************************************")
    print("*************************************************************")
    path = create_folders(args.foldername, args.input_seq_length, args.output_seq_length,args.conditions)
    print("*** Saving results in: ", path, "***")

    print("*** Loading data ***")
    data_train, data_val, train_mask,val_mask = loading_data(feature_list,path)
    
    model, train_loader, val_loader = train(data_train,data_val,train_mask,val_mask,
                                            feature_list = args.feature_list,
                                            net = args.net,
                                            epochs=args.epochs,
                                            batch_size=args.batch_size,
                                            learning_rate=args.learning_rate,
                                            decay=args.decay,
                                            alpha=args.alpha,
                                            gamma=args.gamma,
                                            input_seq_length=args.input_seq_length,
                                            output_seq_length=args.output_seq_length,
                                            output_rnn = args.output_rnn,
                                            num_features=args.num_features,
                                            conditions = args.conditions,
                                            path = path,
                                            missing_values = args.missing_values,
                                            saits_impute = args.saits_impute,
                                            device = device)

    es_callback = EarlyStopping(patience=50, verbose=1, monitor='val_loss', mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',mode="min",
                                          dirpath=path+str('models/'),
                                          filename=str(args.net)+'_{epoch:02d}-{val_loss:.4f}',)
    
    trainer = Trainer(max_epochs=args.epochs,
                      gpus=1 if torch.cuda.is_available() else 0,
                      logger = False,
                      callbacks=[checkpoint_callback,es_callback])

    print("*** Starting training:",args.net,"***")
    trainer.fit(model, train_loader, val_loader)
    
    print(" *** Hparams ***")
    print(model.hparams)

    print("*** Plotting losses ***")
    plotting_losses(path,args.net)

    print("*** Saving parameters ***")
    dict_parameters = {'foldername':path,'epochs':args.epochs,'batch_size':args.batch_size,
                       'learning_rate':args.learning_rate,'decay':args.decay,'net':args.net,
                       'alpha':args.alpha,'gamma':args.gamma,'kfold':args.kfold,
                       'input_seq_length':args.input_seq_length,
                       'output_seq_length':args.output_seq_length,
                       'output_rnn':args.output_rnn,'num_features':args.num_features,
                       'conditions':args.conditions,'missing_values':args.missing_values,
                       'saits_impute':args.saits_impute,'feature_list':args.feature_list}
    with open(path+"models/parameters.txt", 'w') as f: 
        for key, value in dict_parameters.items(): 
            f.write('%s:%s\n' % (key, value))

print("End time:", datetime.datetime.now() - start_time, "Go to folder:", args.foldername)
