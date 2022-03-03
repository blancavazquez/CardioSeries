#python3 bash pipeline.sh

import os
import sys
import json
import pickle
import random
import datetime
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt

#Loading pytorch_lightning
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

#Loading models
from models.cgan import CGAN
from models.gan import GAN
from models.vae import VAE
from models.dcvae import DCVAE
from models.dcgan import DCGAN
from models.seq2seq import encoder, decoder,encoder_decoder
from models.dcnn import DC_CNN_Model 

#Loading losses & utilities
from loss.losses import r2_corr, rmse_loss
from loss.dilate_loss import dilate_loss
from utils import loading_data_secuencial, create_folders, \
                  dwprobability, plotting_predictions, loading_dataloader, \
                  loading_dataloader_conditional, get_data_plus_pid
        
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

start_time = datetime.datetime.now()
print("Start time:",start_time)

def train(train_seqs,train_targets,val_seqs,val_targets,
          database: str,
          net:str,
          output_json_path: str,
          log_dir: str = 'ts_logs',
          model_dir: str = 'ts_models',
          batch_size: int = 10,
          epochs: int = 10,
          learning_rate: float = 3e-4,
          decay: float = 1e-3,
          alpha: float = 1e-3,
          gamma: float = 1e-3,
          window_size: int = 24,
          num_features: int=8,
          feature_list: str = 'temperature'):
    
    print("Epochs:",epochs,"Batch_size:",batch_size)

    if net == "seq2seq":
        enc = encoder(input_size=8, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size)
        dec = decoder(input_size=8, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=8)
        model = encoder_decoder(encoder = enc,decoder = dec,w_decay=decay,dropout=0.1,
                                alpha=alpha,gamma=gamma,target_length=window_size, num_features=num_features,
                                lr = learning_rate).to(device)
        train_loader = loading_dataloader(train_seqs,train_targets,batch_size,"train")
        val_loader = loading_dataloader(val_seqs,val_targets,batch_size,"val")

    elif net == "dcgan":
        model = DCGAN(w_decay=decay,alpha=alpha,gamma=gamma,
                target_length=window_size, num_features=num_features,
                lr = learning_rate).to(device)
        train_loader = loading_dataloader(train_seqs,train_targets,batch_size,"train")
        val_loader = loading_dataloader(val_seqs,val_targets,batch_size,"val")

    elif net == "gan":
        model = GAN(w_decay=decay,alpha=alpha,gamma=gamma,
                target_length=window_size, num_features=num_features,
                lr = learning_rate).to(device)
        train_loader = loading_dataloader(train_seqs,train_targets,batch_size,"train")
        val_loader = loading_dataloader(val_seqs,val_targets,batch_size,"val")

    elif net == "vae":
        model = VAE(train_seqs.shape[-1],w_decay=decay,dropout=0.1,
                alpha=alpha,gamma=gamma,target_length=window_size, 
                num_features=num_features,lr = learning_rate).to(device)
        train_loader = loading_dataloader(train_seqs,train_targets,batch_size,"train")
        val_loader = loading_dataloader(val_seqs,val_targets,batch_size,"val")

    elif net == "dcvae":
        model = DCVAE(train_seqs.shape[-1],w_decay=decay,dropout=0.1,
                alpha=alpha,gamma=gamma,target_length=window_size, 
                num_features=num_features,lr = learning_rate).to(device)
        train_loader = loading_dataloader(train_seqs,train_targets,batch_size,"train")
        val_loader = loading_dataloader(val_seqs,val_targets,batch_size,"val")

    elif net == "dcnn":
        model = DC_CNN_Model(target_length=window_size,num_features=num_features,
                w_decay=decay,dropout=0.1,alpha=alpha,gamma=gamma,lr = learning_rate)
        train_loader = loading_dataloader(train_seqs,train_targets,batch_size,"train")
        val_loader = loading_dataloader(val_seqs,val_targets,batch_size,"val")  
    else:
        sys.exit("You must choose a valid model...") 
        #print("You must choose a valid model...")
    
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
    parser.add_argument("--window_size",type=int,default=24)
    parser.add_argument("--num_features",type=int,default=8)
    parser.add_argument("--conditions",default="age")

    args = parser.parse_args()
    path=create_folders(args.net) #creating folders to save results
    print("* Saving results in:::", path)

    #list for saving results
    list_rmse, list_dilate, list_shape, list_temporal=[],[],[],[]
    list_past,list_pred, list_real,list_r2, list_cond = [],[],[],[],[]
    eed_everything(42) #for reproducibility

    print("---------------------Starting:::",args.net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_list = ['arterial_bp_mean','respiratory_rate', 
                    'diastolic_bp','spo2', 'o2_saturation','heart_rate', 
                    'systolic_bp', 'temperature']

    train_seqs,train_targets,val_seqs,val_targets=loading_data_secuencial(feature_list,args.window_size,path)

    # #**** Ajustando el número de pacientes ****
    train_seqs = train_seqs[:23520] #23520
    train_targets = train_targets[:23520] #23520
    val_seqs = val_seqs[:7840] #7840
    val_targets = val_targets[:7840] #7840

    #For training
    train_seqs= train_seqs.reshape(-1,24,8)
    train_targets= train_targets.reshape(-1,24,8)

    #For validating
    val_seqs= val_seqs.reshape(-1,24,8)
    val_targets= val_targets.reshape(-1,24,8)

    print("train_seqs:",train_seqs.shape,"train_targets:",train_targets.shape)
    print("val_seqs:",val_seqs.shape,"val_targets:",val_targets.shape)


    model, train_loader, val_loader = train(train_seqs,train_targets,
                                                    val_seqs,val_targets,
                                                    database=args.database,
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
                                                    window_size=args.window_size,
                                                    num_features=args.num_features)

    es_callback = EarlyStopping(monitor="loss_val", mode='min', patience=5,verbose=True)
    checkpoint_callback = ModelCheckpoint(monitor="loss_val",mode="min",
                                          dirpath=path+args.model_dir,
                                          filename=str(args.net)+'_{epoch:02d}-{val_loss:.2f}',)

    logger = TensorBoardLogger(save_dir=path+args.log_dir, version=1, name="logs")

    trainer = Trainer(max_epochs=args.epochs,
                      gpus=1 if torch.cuda.is_available() else 0,
                      logger = False,
                      callbacks=[es_callback,checkpoint_callback],
                      progress_bar_refresh_rate=50,
                      gradient_clip_val=0.1)

    if args.net=="vae" or args.net=="seq2seq" or args.net=="dcvae":
        print(" **** Learning rate finder ****")
        lr_finder = trainer.tuner.lr_find(model,train_loader, val_loader,
                                     max_lr=1.0,min_lr=1e-6,early_stop_threshold=None)
        print("LR suggestion:",lr_finder.suggestion()) #all list: lr_finder.results
        model.hparams.lr = lr_finder.suggestion() #fine-tune the learning rate

    print("------------------ Training --------------------------")
    trainer.fit(model, train_loader, val_loader)
    
    print("------------------ Hparams --------------------------")
    print(model.hparams)

    print("----------------- Predictions ------------------------") 
    model.eval()
    with torch.no_grad(): #turn off gradients computation

        for i,data in enumerate(val_loader):
            real_input, real_target = data 

            past = torch.tensor(real_input, dtype=torch.float32).to(device).reshape(-1,args.window_size,args.num_features) #[32, 24, 8])
            expected = torch.tensor(real_target, dtype=torch.float32).to(device).reshape(-1,args.window_size,args.num_features) #[32, 24, 8])

            #---- Prediction ----#
            model = model.to(device)
            if args.net == "seq2seq" or args.net == "gan" or args.net == "dcgan":
                pred = model(past) #[32, 24, 8])
            else: #vae
                pred,_,_ = model(past) #[32, 24, 8])
            #--------------------#

            rmse = rmse_loss(pred,expected)

            loss,loss_shape,loss_temporal = dilate_loss(pred,expected,alpha=args.alpha,
                                                        gamma=args.gamma,device=device)
            pred = pred.reshape(-1,24*8)

            score_R2 = r2_corr(pred.view(-1).detach().cpu().numpy().reshape(-1,1),
                               expected.view(-1).detach().cpu().numpy().reshape(-1,1))

            list_rmse.append(rmse.item())
            list_dilate.append(loss.item())
            list_shape.append(loss_shape.item())
            list_temporal.append(loss_temporal.item())
            list_r2.append(score_R2) 

            list_past.extend(past.view(-1).detach().cpu().numpy().reshape(-1,1))
            list_real.extend(expected.view(-1).detach().cpu().numpy().reshape(-1,1))
            list_pred.extend(pred.view(-1).detach().cpu().numpy().reshape(-1,1))

    print("RMSE:", round(np.mean(list_rmse),3))
    print("Dilate:", round(np.mean(list_dilate),3))
    print("Shape:", round(np.mean(list_shape),3))
    print("Temporal:", round(np.mean(list_temporal),3))
    print("R2:", round(np.mean(list_r2),3))

    dwprobability(list_real,list_pred,args.window_size,args.num_features,path, args.net)

    plotting_predictions(list_past,list_real,list_pred,args.window_size,args.num_features,feature_list,path,
                         args.net,list_cond)

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
