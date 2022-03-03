#source: https://github.com/jamesloyys/PyTorch-Lightning-GAN/blob/main/GAN/gan.py

#https://github.com/TeeyoHuang/conditional-GAN/blob/master/conditional_gan.py (lineal)

import numpy as np
import torch
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
      torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm1d') != -1:
      torch.nn.init.normal(m.weight, 1.0, 0.02)
      torch.nn.init.constant(m.bias, 0.0)

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
    #[N,2,97] => [N,8,32]
    self.cnn_encoder = nn.Sequential(
      nn.Conv1d(in_channels=2,out_channels=8,kernel_size=4,stride=3,padding=0,bias=False),
      nn.BatchNorm1d(8),
      nn.Dropout(p=0.25),
      nn.LeakyReLU(0.2, inplace=True))

    #------- Decoder -----------#
    # [N, 8,32] => [N, 256, 32]
    self.cnn_block_1 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=8,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
      nn.BatchNorm1d(256, momentum=0.1,  eps=0.8),
      nn.ReLU(True))

    # [N, 256, 32] => [N, 128, 64]
    self.cnn_block_2 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(128, momentum=0.1,  eps=0.8),
      nn.ReLU(True))

    # [N, 128, 64] => [N, 64, 128]
    self.cnn_block_3 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(64, momentum=0.1,  eps=0.8),
      nn.ReLU(True))

    # [N, 64, 128] => [N, 1, 256]
    self.cnn_block_4 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=64,out_channels=1,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(1, momentum=0.1,  eps=0.8),
      nn.ReLU(0.2))

    # [N,1,256] => [32,1,1024]
    self.hidden_layer2 = nn.Sequential(nn.Linear(256, 1024),nn.ReLU(0.2),nn.Dropout(p=0.25),)

    # [N,1,1024] => [32,1,192]
    self.hidden_layer3 = nn.Sequential(nn.Linear(1024, output_dim),nn.Tanh())

  def forward(self, x_input,cond): #noise=[32,24,8],y=[32,2]
    x_input_np = x_input.reshape(-1,x_input.shape[1]*x_input.shape[2]) #[32, 192]
    cond = cond.long() #[32,2] 
    x = torch.cat([x_input_np,cond],1) #[32,194]
    output = x.reshape(-1,2,97) #[32,2,97]
    output = self.cnn_encoder(output) #[32,8,32]
    output = self.cnn_block_1(output) #[32,256,32]
    output = self.cnn_block_2(output) #[32,128,64]
    output = self.cnn_block_3(output) #[32,64,128]
    output = self.cnn_block_4(output) #[32,1,256]
    output = self.hidden_layer2(output) #[32,1,1024]
    output = self.hidden_layer3(output) #[32,1,192]
    output = output.reshape(-1,x_input.shape[1],x_input.shape[2]) #[32, 24,8]

    #print("salida G:",output[1])
    return output

class Discriminator(nn.Module):
  '''
  Input: [32,24,8]
  Output:[768,1]
  '''

  def __init__(self,
               batch_size=32,
               seq_length=24,
               num_features=8):
    super().__init__()
    input_dim = 192+2
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

    # [N,1,256] => [32,192]
    self.hidden_layer1 = nn.Sequential(nn.Linear(192, 192), nn.Dropout(p=0.25),nn.LeakyReLU(0.2, inplace=True))

    # [N,1,256] => [32,192]
    self.hidden_layer2 = nn.Sequential(nn.Linear(192, 1))#,nn.Tanh())
        
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


class WCGAN(pl.LightningModule):
  def __init__(self,
               lr=3e-4,
               w_decay=1e-4,
               alpha=1e-4,
               gamma=1e-4,
               target_length = 24,
               num_features = 8):
    super().__init__()
    self.generator = Generator()
    self.discriminator = Discriminator()
    self.lr = lr
    self.w_decay = w_decay
    self.alpha = alpha
    self.gamma = gamma
    self.target_length = target_length
    self.num_features = num_features
    self.save_hyperparameters()

    # Initialize weights
    self.generator.apply(weights_init_normal)
    self.discriminator.apply(weights_init_normal)

  def forward(self, z,y):
    """ Generates an image using the generator given input noise z and labels y """
    return self.generator(z,y)

  def generator_step(self,x_input,conditions):
    """
    x_input[32,24,8]  (first 24 hours)
    cond[32,2] (sex,age)
    """
    
    # Generate images
    generated_ehr = self(x_input,conditions) #[32,24,8]

    # Classify generated images
    d_output = torch.squeeze(self.discriminator(generated_ehr,conditions)) #[32]

    # Backprop loss
    g_loss = -torch.mean(d_output)

    return g_loss

  def discriminator_step(self,x_input,x_target,conditions):
    """
    x_input[32,24,8]  (first 24 hours)
    x_target[32,24,8]  (next 24 hours)
    cond[32,2] (sex,age)
    
    Training step for discriminator
    1. Get actual images
    2. Get fake images from generator
    3. Predict probabilities of actual images
    4. Predict probabilities of fake images
    5. Get loss of both and backprop
    """
    #-------Get real EHR-------#
    d_output = torch.squeeze(self.discriminator(x_target,conditions)) #[32]
    
    loss_real = -torch.mean(d_output)

    #-------Get fake EHR-------#
    generated_ehr = self(x_input,conditions) #[32,24,8]

    d_output = torch.squeeze(self.discriminator(generated_ehr,conditions)) #[32]

    loss_fake = -torch.mean(d_output)

    d_loss = -(torch.mean(loss_real) - torch.mean(loss_fake))

    return d_loss

  def training_step(self, batch, batch_idx, optimizer_idx):
    target_in, target_out,condition = batch #in [32,24,8], out [32,24,8], cond [32,2]

    clip_value = 0.01

    if optimizer_idx == 0: ## Generator
      loss = self.generator_step(target_in,condition)
      self.log("g_loss", loss)
      #print("G_loss::", loss.item())
      
    if optimizer_idx == 1: ## Discriminator
      loss = self.discriminator_step(target_in,target_out,condition)      
      for p in self.discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

      self.log("d_loss", loss)
    
    self.log("loss_train",loss)
    return loss

  def validation_step(self, batch, batch_idx):
    target_in, target_out,condition = batch #in [32,24,8], out [32,24,8], cond [32,2]      

    loss = self.discriminator_step(target_in,target_out,condition)
    self.log("loss_val",loss)
    return loss

  def configure_optimizers(self):
    g_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=5e-5)
    d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-5)
    return [g_optimizer, d_optimizer],[]

