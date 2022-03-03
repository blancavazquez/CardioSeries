#GRU VAE::: https://discuss.pytorch.org/t/gru-time-series-autoencoder/77126

"""
THE ENCODER COMPRESSES DATA INTO A LATENT SPACE (Z). 
THE DECODER RECONSTRUCTS THE DATA GIVEN THE HIDDEN REPRESENTATION.
"""
import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import pytorch_lightning as pl
from loss.dilate_loss import dilate_loss
from loss.losses import r2_corr,rmse_loss,mmd_loss, vae_loss

class encoder(torch.nn.Module):
    def __init__(self,input_size):
        super(encoder, self).__init__() #(N,C,L)
        print("input_size:::", input_size) #()
        self.conv1 = nn.Conv1d(in_channels=24,out_channels=input_size,kernel_size=4,padding=1,stride=1)
        self.conv2 = nn.Conv1d(in_channels=input_size,out_channels=4,kernel_size=4,padding=1,stride=1)
        self.fc_mu = nn.Linear(in_features=24,out_features=10)
        self.fc_logvar = nn.Linear(in_features=24, out_features=10)


    def forward(self, x):    # its output is a hidden representation z
        x = F.relu(self.conv1(x)) #([32, 8, 7])
        x = F.relu(self.conv2(x)) #([32, 4, 6])
        x = x.view(x.size(0),-1)# ([32, 24]) flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x) #([32, 10])
        x_logvar = self.fc_logvar(x) #([32, 10])
        return x_mu, x_logvar
    
class decoder(torch.nn.Module):
    def __init__(self, input_size):
        super(decoder, self).__init__()
        self.fc = nn.Linear(in_features=10,out_features=24)
        self.conv2 = nn.ConvTranspose1d(in_channels=4,out_channels=input_size,kernel_size=4,padding=1,stride=1)
        self.conv1 = nn.ConvTranspose1d(in_channels=input_size,out_channels=24,kernel_size=4,padding=1,stride=1) #2--3
            
    def forward(self, x):
        x = self.fc(x) #([32, 24])
        x = x.view(x.size(0),4,6)# ([32, 4, 6]) unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x)) #([32, 8, 7])
        x = torch.sigmoid(self.conv1(x))# ([32, 24, 8]) last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x

    
class CNN_VAE(pl.LightningModule):
    def __init__(self, input_size,w_decay,
                 dropout,alpha,gamma,target_length,num_features,lr):
        super(CNN_VAE, self).__init__()
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

    def latente(self,x):
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.size(), device=mu.device)
        z = mu + eps * std
        return z

    def forward(self,x):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        mu, logvar = self.encoder(x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn(mu.size(),device=mu.device)
        z = mu + eps * std
        xrec = self.decoder(z) #([32, 24, 8])
        return xrec, mu, logvar
                
    def training_step(self, batch, batch_idx):
        inputs,targets = batch # inputs:[32, 24, 8], #targets:[32, 24, 8]

        inputs = torch.tensor(inputs, dtype=torch.float32).to(inputs.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(inputs.device)
        
        pred,mu,logvar = self(inputs) #([32, 24, 8]),  ([32, 10]),  ([32, 10])
        loss = vae_loss(pred,targets,mu,logvar)
        self.log("train_loss", loss)
        return loss

    def validation_step(self,batch,batch_idx):
        inputs,targets = batch # inputs:[32, 24, 8], #targets:[32, 24, 8]

        inputs = torch.tensor(inputs, dtype=torch.float32).to(inputs.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(inputs.device)
        
        pred,mu,logvar = self(inputs) #([32, 24, 8]),  ([32, 10]),  ([32, 10])
        loss = vae_loss(pred,targets,mu,logvar)
        self.log("val_loss",loss)

        return loss

    def test_step(self,batch,batch_idx): #recorre TODO el val_set por batch
        inputs,targets = batch

        inputs = torch.tensor(inputs, dtype=torch.float32).to(inputs.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(inputs.device)

        y_pred,_,_ = self(inputs)
        loss,loss_shape,loss_temporal = dilate_loss(y_pred,targets,alpha=self.alpha,
                                                    gamma=self.gamma,device=inputs.device)

        loss_rmse = rmse_loss(y_pred,y_real)

        self.log("loss_dilate",loss)
        self.log("loss_shape",loss_shape)
        self.log("loss_temporal",loss_temporal)
        self.log("loss_rmse",loss_rmse)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.w_decay)        
        return {"optimizer": optimizer,"monitor": "loss",}      