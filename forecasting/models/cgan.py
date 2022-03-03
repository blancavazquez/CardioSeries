#source: https://github.com/jamesloyys/PyTorch-Lightning-GAN/blob/main/GAN/gan.py

#https://github.com/TeeyoHuang/conditional-GAN/blob/master/conditional_gan.py (lineal)

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict
from loss.dilate_loss import dilate_loss
from loss.losses import r2_corr,rmse_loss,mmd_loss

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
      torch.nn.init.normal(m.weight, mean = 0.0, std = 0.02)
    elif classname.find('BatchNorm1d') != -1:
      torch.nn.init.normal(m.weight, mean = 1.0, std = 0.02)
      torch.nn.init.constant(m.bias, 0.0)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, ypred, yreal, smooth=1):
      #comment out if your model contains a sigmoid or equivalent activation layer
      #ypred = F.sigmoid(ypred)       
      
      #flatten label and prediction tensors
      ypred = ypred.view(-1)
      yreal = yreal.view(-1)
      
      intersection = (ypred * yreal).sum()

      dice_loss = 1 - (2.*intersection + smooth)/(ypred.sum() + yreal.sum() + smooth) 

      BCE = F.binary_cross_entropy(ypred, yreal, reduction='mean')
      
      Dice_BCE = BCE + dice_loss
      
      return Dice_BCE

class Generator(nn.Module):
  def __init__(self,
               batch_size=32,
               seq_length=24,
               num_features=8):
    super().__init__()
    input_dim = 192+2
    output_dim = 192
    # N = batch_size

    #------- Encoder -----------#
    #[N,2,97] => [N,8,24]
    self.cnn_encoder = nn.Sequential(
      nn.Conv1d(in_channels=1,out_channels=8,kernel_size=4,stride=4,padding=1,bias=False),
      nn.BatchNorm1d(8,momentum=0.3,  eps=0.8),
      nn.Dropout(p=0.25),
      nn.LeakyReLU(0.2, inplace=True))

    #------- Decoder -----------#
    # [N, 8,24] => [N, 256, 8]
    self.cnn_block_1 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=8,out_channels=256,kernel_size=1,stride=1,padding=8,bias=False),
      nn.BatchNorm1d(256, momentum=0.3,  eps=0.8),
      nn.ReLU(inplace=True))

    # [N, 256, 8] => [N, 128, 16]
    self.cnn_block_2 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(128, momentum=0.3,  eps=0.8),
      nn.ReLU(inplace=True))

    # [N, 128, 64] => [N, 64, 32]
    self.cnn_block_3 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(64, momentum=0.3,  eps=0.8),
      nn.ReLU(inplace=True))

    # [N, 64, 256] => [N, 1, 64]
    self.cnn_block_4 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(32, momentum=0.3,  eps=0.8),
      nn.ReLU(inplace=True))

    # [N, 1, 64] => [N, 1, 192]
    self.cnn_block_5 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=32,out_channels=1,kernel_size=3,stride=3,padding=0,bias=False),
      nn.BatchNorm1d(1, momentum=0.3,  eps=0.8),
      nn.Tanh())

    # [N,1,256] => [32,1,1024]
    ### En caso de probar con más o menos condiciones, cambiar el "792"
    self.hidden_layer1 = nn.Sequential(nn.Linear(792, 512),nn.ReLU(0.2),nn.Dropout(p=0.25),)

    # [N,1,1024] => [32,1,192]
    self.hidden_layer2 = nn.Sequential(nn.Linear(512, output_dim),nn.Tanh())

  def forward(self, x_input,cond): #noise=[32,24,8],y=[32,2]
    x_input_np = x_input.reshape(-1,x_input.shape[1]*x_input.shape[2]) #[32, 192]
    cond = cond.long() #[32,2] 
    x = torch.cat([x_input_np,cond],1) #[32,194]
    
    cond_plus_data = x_input.shape[1]*x_input.shape[2]+ cond.shape[1]

    output = x.reshape(-1,1,cond_plus_data) #[32,2,97]

    output = self.cnn_encoder(output) #[32,8,24]
    #print("cnn_encoder", output.shape)

    output = self.cnn_block_1(output) #[32,256,8]
    #print("cnn_block_1", output.shape)

    output = self.cnn_block_2(output) #[32,128,16]
    #print("cnn_block_2", output.shape)

    output = self.cnn_block_3(output) #[32,64,32]
    #print("cnn_block_3", output.shape)

    output = self.cnn_block_4(output) #[32,32,64]
    #print("cnn_block_4", output.shape)

    output = self.cnn_block_5(output) #[32,1,192]
    #print("cnn_block_5", output.shape)
    
    output = self.hidden_layer1(output) #[32,1,192]
    #print("hidden_layer1", output.shape)

    output = self.hidden_layer2(output) #[32,1,192]
    #print("hidden_layer2", output.shape)

    output = output.reshape(-1,x_input.shape[1],x_input.shape[2]) #[32, 24,8]
    return output

class Discriminator(nn.Module):
  def __init__(self,target_length,num_features,num_conditions):
    super(Discriminator,self).__init__()

    self.target_length = target_length
    self.num_features = num_features
    self.num_conditions = num_conditions
    
    input_dim = self.target_length*num_features + num_conditions
    output_dim = 1

    # [N, 1, 194] => [N, 64, 192]
    self.cnn_block_1 = nn.Sequential(
      nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=0,bias=False),
      nn.BatchNorm1d(64),
      nn.Dropout(p=0.25),
      nn.LeakyReLU(0.2, inplace=True),
      )

    # [N, 64, 192] => [N,128,192]
    self.cnn_block_2 = nn.Sequential(
      nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False),
      nn.BatchNorm1d(128),
      nn.Dropout(p=0.25),
      nn.LeakyReLU(0.2, inplace=True))

    # [N, 128, 192] => [N,256,192]
    self.cnn_block_3 = nn.Sequential(
      nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
      nn.BatchNorm1d(256),
      nn.Dropout(p=0.25),
      nn.LeakyReLU(0.2, inplace=True))

    # [N, 256, 192] => [N,192]
    self.cnn_block_4 = nn.Sequential(
      nn.Conv1d(in_channels=256,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False),
      nn.Dropout(p=0.25),
      nn.Flatten())

    # ### En caso de probar con más o menos condiciones, cambiar el "195"
    self.hidden_layer1 = nn.Sequential(nn.Linear(195,192), nn.Dropout(p=0.25),nn.LeakyReLU(0.2, inplace=True))

    # [N,1,256] => [32,192]
    self.hidden_layer2 = nn.Sequential(nn.Linear(192, output_dim),nn.Sigmoid())
        
  def forward(self, x,cond): #x=[32,24,8],cond=[32,2]
    x = x.reshape(-1,24*8) #[32, 192]
    cond = cond.long() #[32,2]
    x = torch.cat([x,cond],1) #[32,194]
    x = x.reshape(-1,1,x.shape[1]) #[32,1,194]
    output = self.cnn_block_1(x) #[32,64,192]
    output = self.cnn_block_2(output) #[32,128,192]
    output = self.cnn_block_3(output) #[32,256,192]
    output = self.cnn_block_4(output) #[32,192]
    output = self.hidden_layer1(output) #[32,192]
    output = self.hidden_layer2(output) #[32,1]

    return output


class CGAN(pl.LightningModule):
  def __init__(self,
               lr=3e-4,w_decay=1e-4,alpha=1e-4,gamma=1e-4,
               target_length = 24,
               num_features = 8,
               num_conditions = 2):
    super().__init__()
    self.generator = Generator()
    self.discriminator = Discriminator(target_length,num_features,num_conditions)
    self.DiceBCELoss=DiceBCELoss()
    self.lr = lr
    self.w_decay = w_decay
    self.alpha = alpha
    self.gamma = gamma
    self.target_length = target_length
    self.num_features = num_features
    self.num_conditions = num_conditions
    self.save_hyperparameters()

    # Initialize weights
    self.generator.apply(weights_init_normal)
    self.discriminator.apply(weights_init_normal)

  def forward(self, z,y):
    """ Generates an EHR using the generator given first 24 hours and conditions """
    return self.generator(z,y)

  def generator_step(self,x_input,conditions):
    """
    x_input[32,24,8]  (first 24 hours)
    cond[32,2] (sex,age)
    """

    # Generate EHR
    generated_ehr = self(x_input,conditions) #[32,24,8]

    # Classify generated EHR
    d_output = torch.squeeze(self.discriminator(generated_ehr,conditions)) #[32]

    #label
    y_true = torch.ones(x_input.shape[0], device=self.device) #[32]

    # Backprop loss
    #g_loss = nn.BCELoss()(d_output, y_true)

    g_loss = rmse_loss(d_output, y_true)

    return g_loss

  def discriminator_step(self,x_input,x_target,conditions):
    """
    Measure discriminator's ability to classify real from generated samples
    x_input[32,24,8]  (first 24 hours)
    x_target[32,24,8]  (next 24 hours)
    cond[32,2] (sex,age)
    """

    #-------Get real EHR-------#
    d_output = torch.squeeze(self.discriminator(x_target,conditions)) #[32]

    y_true = torch.ones(x_target.shape[0], device=self.device) #[32]

    #y_true = y_true - 0.3 + (torch.rand(y_true.shape,device=self.device) * 0.5)
    #loss_real = nn.BCELoss()(d_output,y_true) #predict_loss_real

    loss_real = DiceBCELoss()(d_output, y_true)

    #-------Get fake EHR-------#
    generated_ehr = self(x_input,conditions) #[32,24,8]

    d_output = torch.squeeze(self.discriminator(generated_ehr,conditions)) #[32]

    y_fake = torch.zeros(x_input.shape[0], device=self.device)#[32]

    #loss_fake = nn.BCELoss()(d_output, y_fake)#predict_loss_fake
    #d_loss = (loss_real + loss_fake)/2

    #---------------- New loss function ----------------#
    dilate, loss_shape, loss_temporal = dilate_loss(generated_ehr, x_target, alpha=self.alpha,gamma=self.gamma,device=self.device)
    #new_loss = 1 - (loss_real*0.5 + loss_fake*0.03  + (loss*0.002+ loss_shape*0.002+loss_temporal**0.002))

    loss_fake = DiceBCELoss()(d_output, y_fake)

    d_loss = 1 - (loss_real + loss_fake)/2 + (dilate*0.008)

    return  d_loss

  def training_step(self, batch, batch_idx, optimizer_idx):
    target_in, target_out,condition = batch #in [32,24,8], out [32,24,8], cond [32,2]

    if optimizer_idx == 0: ## Generator
      loss = self.generator_step(target_in,condition)
      self.log("g_loss", loss)
      #print("G_loss::", loss.item())
      
    if optimizer_idx == 1: ## Discriminator
      loss = self.discriminator_step(target_in,target_out,condition)
      self.log("d_loss", loss)
      #print("D_loss::", loss.item())
    
    self.log("loss_train",loss)
    return loss

  def validation_step(self, batch, batch_idx):
    target_in, target_out,condition = batch #in [32,24,8], out [32,24,8], cond [32,2]

    loss = self.discriminator_step(target_in,target_out,condition)
    self.log("loss_val",loss)

    #añadir
    #output = dict({'loss_val': loss,})

    return loss #output

  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr,weight_decay=self.w_decay,betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr,weight_decay=self.w_decay,betas=(0.5, 0.999))
    return [g_optimizer, d_optimizer],[]

