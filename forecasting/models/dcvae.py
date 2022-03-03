#https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=9xb-hMWBNsbB

#VAE in deep: https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
#https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

#cnn vae: https://github.com/sksq96/pytorch-vae
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
        super(encoder, self).__init__() #(N,C,L)
        input_dim = 192
        output_dim = 192

        # [N,192] =>[32,256]
        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim, 256),
                                           nn.BatchNorm1d(256),
                                           nn.ReLU(0.2))
        # [N, 8,32] => [N, 256, 32]
        self.cnn_block_1 = nn.Sequential(
          nn.Conv1d(in_channels=8,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
          nn.BatchNorm1d(256, momentum=0.1,  eps=0.8),
          nn.ReLU(0.2))

        # [N, 256, 32] => [N, 128, 16]
        self.cnn_block_2 = nn.Sequential(
          nn.Conv1d(in_channels=256,out_channels=128,kernel_size=3,stride=2,padding=1,bias=False),
          nn.BatchNorm1d(128, momentum=0.1,  eps=0.8),
          nn.ReLU(0.2))

        # [N, 128, 16] => [N, 64, 8]
        self.cnn_block_3 = nn.Sequential(
          nn.Conv1d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
          nn.BatchNorm1d(64, momentum=0.1,  eps=0.8),
          nn.ReLU(0.2))

        # [N, 64, 8] => [N, 1, 4]
        self.cnn_block_4 = nn.Sequential(
          nn.Conv1d(in_channels=64,out_channels=1,kernel_size=3,stride=2,padding=1,bias=False),
          nn.BatchNorm1d(1, momentum=0.1,  eps=0.8),
          nn.ReLU(0.2))

        #[N,4] => [32,32]
        self.fc_mu = nn.Linear(in_features=4,out_features=32)

        #[N,4] => [32,32]
        self.fc_logvar = nn.Linear(in_features=4, out_features=32)


    def forward(self, x_input):    # its output is a hidden representation z
        output = x_input.reshape(-1,x_input.shape[1]*x_input.shape[2]) #[32,192]
        output = self.hidden_layer1(output) #[32,256]
        output = output.reshape(-1,8,32) #[32,8,32]
        output = F.relu(self.cnn_block_1(output)) #([32, 256,32])
        output = F.relu(self.cnn_block_2(output)) #([32, 128, 16])
        output = F.relu(self.cnn_block_3(output)) #([32, 64, 8])
        output = F.relu(self.cnn_block_4(output)) #([32, 1, 4]
        output = output.view(output.size(0),-1)# ([32, 4]) flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(output) #([32, 32])
        x_logvar = self.fc_logvar(output) #([32, 32])
        return x_mu, x_logvar
    
class decoder(torch.nn.Module):
    def __init__(self, input_size):
        super(decoder, self).__init__()
        # [N, 1, 32] => [N, 64, 32]
        self.cnn_block_1 = nn.Sequential(
          nn.ConvTranspose1d(in_channels=1,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
          nn.BatchNorm1d(256),
          nn.Dropout(p=0.25),
          nn.LeakyReLU(0.2, inplace=True),
          )

        # [N, 64, 32] => [N,128,32]
        self.cnn_block_2 = nn.Sequential(
          nn.ConvTranspose1d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False),
          nn.BatchNorm1d(128),
          nn.Dropout(p=0.25),
          nn.LeakyReLU(0.2, inplace=True))

        # [N, 128, 32] => [N,256,32]
        self.cnn_block_3 = nn.Sequential(
          nn.ConvTranspose1d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
          nn.BatchNorm1d(64),
          nn.Dropout(p=0.25),
          nn.LeakyReLU(0.2, inplace=True))

        # [N, 256, 32] => [N,32]
        self.cnn_block_4 = nn.Sequential(
          nn.ConvTranspose1d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False),
          nn.Dropout(p=0.25),
          nn.Flatten())

        # [N,32] => [N,192]
        self.hidden_layer1 = nn.Sequential(nn.Linear(32, 192), nn.Dropout(p=0.25),nn.Sigmoid())
            
    def forward(self, x_input): #[32,32]        
        output = x_input.reshape(-1,1,x_input.shape[1]) #[32,1,32]
        output = F.relu(self.cnn_block_1(output)) #([32, 256,32])
        output = F.relu(self.cnn_block_2(output)) #([32, 128, 32])
        output = F.relu(self.cnn_block_3(output)) #([32, 64, 32])
        output = F.relu(self.cnn_block_4(output)) #([32,32])
        output = F.relu(self.hidden_layer1(output)) #([32,192])
        output = output.reshape(-1,24,8) #[32,24,8] last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return output

    
class DCVAE(pl.LightningModule):
    def __init__(self, input_size,w_decay,
                 dropout,alpha,gamma,target_length,num_features,lr):
        super(DCVAE, self).__init__()
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
        mu, logvar = self.encoder(x) #mu [32,32], logvar [32,32]
        std = torch.exp(0.5*logvar)
        eps = torch.randn(mu.size(),device=mu.device)
        z = mu + eps * std #[32,32]
        xrec = self.decoder(z) #([32, 24, 8])
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
        #print("loss_val:", np.round(loss.item(),3))
        
        self.log("loss_val",loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.w_decay,betas=(0.5, 0.999))        
        return {"optimizer": optimizer,"monitor": "loss",}      