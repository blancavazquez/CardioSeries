#source: https://github.com/gibranfp/CursoAprendizajeProfundo/blob/2022-1/notebooks/6c_vae.ipynb

#VAE in deep: https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
#https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

"""
THE ENCODER COMPRESSES DATA INTO A LATENT SPACE (Z). 
THE DECODER RECONSTRUCTS THE DATA GIVEN THE HIDDEN REPRESENTATION.
"""
import numpy as np
import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import pytorch_lightning as pl
from loss.dilate_loss import dilate_loss
from loss.losses import r2_corr,rmse_loss,mmd_loss, vae_loss

class encoder(torch.nn.Module):
    def __init__(self,input_size):
        super(encoder, self).__init__() #Its input is a datapoint xx
        #[N,192]=>[32,32]
        self.densa_in = nn.Sequential(nn.Linear(192,1024),nn.BatchNorm1d(1024),nn.ReLU(0.2),
                                      nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(0.2),
                                      nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(0.2),
                                      nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(0.2),
                                      nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(0.2),
                                      nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(0.2),)
        self.densa_mu = nn.Linear(32,32)
        self.densa_logvar = nn.Linear(32,32)  

    def forward(self, x_input): #[32,24,8]    # its output is a hidden representation z
        x = x_input.reshape(-1,x_input.shape[1]*x_input.shape[2]) #[32,192]
        h = self.densa_in(x) #[32,32]
        mu = self.densa_mu(h) #[32,32]
        logvar = self.densa_logvar(h) #[32,32]
        return mu, logvar
    
class decoder(torch.nn.Module):
    def __init__(self, input_size):
        super(decoder, self).__init__()
    
        #[N,32]=>[32,24,8]
        self.red_densa = nn.Sequential(nn.Linear(32,64),nn.BatchNorm1d(64),nn.LeakyReLU(0.2),
                                       nn.Linear(64,128),nn.BatchNorm1d(128),nn.LeakyReLU(0.2),
                                       nn.Linear(128,256),nn.BatchNorm1d(256),nn.LeakyReLU(0.2),
                                       nn.Linear(256,512),nn.BatchNorm1d(512),nn.LeakyReLU(0.2),
                                       nn.Linear(512,1024),nn.BatchNorm1d(1024),nn.LeakyReLU(0.2),
                                       nn.Linear(1024,192),nn.Sigmoid())    
        
    def forward(self, z): #[32,32]
        x = self.red_densa(z) #[32,192]
        x = x.reshape(-1,24,8) #[32, 24, 8]
        return x
    
class VAE(pl.LightningModule):
    def __init__(self, input_size,w_decay,
                 dropout,alpha,gamma,target_length,num_features,lr):
        super(VAE, self).__init__()
        self.encoder = encoder(input_size)
        self.decoder = decoder(input_size)
        self.w_decay = w_decay
        self.dropout = dropout
        self.alpha = alpha
        self.gamma = gamma
        self.target_length = target_length
        self.num_features = num_features
        self.lr = lr
        self.save_hyperparameters()

    def latente(self,x): #???
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.size(), device=mu.device)
        z = mu + eps * std
        return z

    def forward(self,x):
        mu, logvar = self.encoder(x) #mu [32,32], logvar [32,32]
        std = torch.exp(0.5*logvar)
        eps = torch.randn(mu.size(),device=mu.device)
        z = mu + eps * std #[32,32]
        xrec = self.decoder(z) #[32,24,8]
        return xrec, mu, logvar
                
    def training_step(self, batch, batch_idx):
        target_in, target_out= batch #in [32,24,8], out [32,24,8]

        target_in = torch.tensor(target_in, dtype=torch.float32)
        target_out = torch.tensor(target_out, dtype=torch.float32)

        pred,mu,logvar = self(target_in) #([32, 24, 8]),([32, 24, 50]),([32, 24, 50])

        loss = vae_loss(pred,target_out,mu,logvar)

        self.log("loss_train", loss)
        return loss


    def validation_step(self,batch,batch_idx):
        target_in, target_out= batch #in [32,24,8], out [32,24,8]

        target_in = torch.tensor(target_in, dtype=torch.float32)
        target_out = torch.tensor(target_out, dtype=torch.float32)

        pred,mu,logvar = self(target_in) #([32, 24, 8]),([32, 24, 50]),([32, 24, 50])

        loss = vae_loss(pred,target_out,mu,logvar)
        print("loss_val:", np.round(loss.item(),3))
        
        self.log("loss_val",loss)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.w_decay,betas=(0.5, 0.999))        
        return {"optimizer": optimizer,"monitor": "loss",}      