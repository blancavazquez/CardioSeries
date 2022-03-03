#fuente GAN: https://github.com/jamesloyys/PyTorch-Lightning-GAN/blob/main/GAN/gan.py

"""
THE ENCODER COMPRESSES DATA INTO A LATENT SPACE (Z). 
THE DECODER RECONSTRUCTS THE DATA GIVEN THE HIDDEN REPRESENTATION.
"""

import torch
import torch.nn as nn
from math import sqrt
from collections import OrderedDict
import torch.nn.functional as F
import pytorch_lightning as pl
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
    #[N,192] => [32,256]
    self.layer1 = nn.Sequential(nn.Linear(192,256),nn.BatchNorm1d(256),nn.ReLU(0.2))
    #[N,256] => [N,512]
    self.layer2 = nn.Sequential(nn.Linear(256,512),nn.BatchNorm1d(512),nn.ReLU(0.2))
    #[N,512] => [N,1024]
    self.layer3 = nn.Sequential(nn.Linear(512,1024),nn.BatchNorm1d(1024),nn.ReLU(0.2))
    #[N,1024] => [N,192]
    self.layer4 = nn.Sequential(nn.Linear(1024,192),nn.Tanh()) #mat1=768x8

  def forward(self, x_input): #[32,24,8]
    output= x_input.reshape(-1,x_input.shape[1]*x_input.shape[2])#[32,192]
    output = self.layer1(output) #[32,256]
    output = self.layer2(output)#[32,512]
    output = self.layer3(output)#[32,1024]
    output = self.layer4(output) #[32,192]
    output = output.reshape(-1,x_input.shape[1],x_input.shape[2]) #[32, 24,8]
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
    #[N,192] => [32,1024]
    self.layer1 = nn.Sequential(nn.Linear(192, 1024),nn.BatchNorm1d(1024),nn.LeakyReLU(0.2, inplace=True)) #8
    #[N,192] => [32,512]
    self.layer2 = nn.Sequential(nn.Linear(1024, 512),nn.BatchNorm1d(512),nn.LeakyReLU(0.2, inplace=True))
    #[N,512] => [32,256]
    self.layer3 = nn.Sequential(nn.Linear(512, 256),nn.BatchNorm1d(256),nn.LeakyReLU(0.2, inplace=True))
    #[N,256] => [32,1]
    self.layer4 = nn.Sequential(nn.Linear(256, 1),nn.Sigmoid()) #8
    
  def forward(self, x_input):
    output = x_input.reshape(-1,x_input.shape[1]*x_input.shape[2]) #[32,192]
    output = self.layer1(output) #[32,1024]
    output = self.layer2(output)#[32,512]
    output = self.layer3(output) #[32,256]
    output = self.layer4(output) #[32,1]
    return output


class GAN(pl.LightningModule):
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


  def forward(self, z):
    """Generates an image using the generator given input noise z"""
    return self.generator(z)

  def generator_step(self, x_input):
    """
    x_input[32,24,8]  (first 24 hours)
    """
    
    # Generate EHR
    generated_ehr = self(x_input) #[32,24,8]

    # Classify generated EHR
    d_output = torch.squeeze(self.discriminator(generated_ehr)) #[32]

    #label
    y_true = torch.ones((x_input.shape[0]), device=self.device) #[32]

    # Backprop loss
    g_loss = nn.BCELoss()(d_output, y_true)

    return g_loss

  def discriminator_step(self, x_input,x_target,):
    """
    Measure discriminator's ability to classify real from generated samples
    x_input[32,24,8]  (first 24 hours)
    x_target[32,24,8]  (next 24 hours)
    """
    
    #-------Get real EHR-------#
    d_output = torch.squeeze(self.discriminator(x_target)) #[32]

    y_true = torch.ones(x_target.shape[0], device=self.device) #[32]

    loss_real = nn.BCELoss()(d_output,y_true) #predict_loss_real
  
    #-------Get fake EHR-------#
    generated_ehr = self(x_input) #[32,24,8]
    
    d_output = torch.squeeze(self.discriminator(generated_ehr)) #[32]
    
    y_fake = torch.zeros(x_input.shape[0], device=self.device)#[32]

    loss_fake = nn.BCELoss()(d_output, y_fake) #predict_loss_fake
    
    d_loss = (loss_real + loss_fake)/2

    return d_loss

  def training_step(self, batch, batch_idx, optimizer_idx):
    target_in, target_out = batch #target_in: [32, 24, 9], trg_out: [32, 24, 8]

    target_in = torch.tensor(target_in, dtype=torch.float32).to(target_in.device)
    target_out = torch.tensor(target_out, dtype=torch.float32).to(target_out.device)

    if optimizer_idx == 0:
      loss = self.generator_step(target_in)
      self.log("g_loss", loss)

    if optimizer_idx == 1:
      loss = self.discriminator_step(target_in, target_out)
      self.log("d_loss", loss)
    
    return loss

  def validation_step(self, batch, batch_idx):
    target_in, target_out = batch #target_in: [32, 24, 9], trg_out: [32, 24, 8]

    target_in = torch.tensor(target_in, dtype=torch.float32).to(target_in.device)
    trg_out = torch.tensor(target_out, dtype=torch.float32).to(target_out.device)

    loss = self.discriminator_step(target_in, trg_out)
    print("loss_val", loss.item())
    
    self.log("loss_val", loss)
    return loss

  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr,weight_decay=self.w_decay,betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr,weight_decay=self.w_decay,betas=(0.5, 0.999))
    return [g_optimizer, d_optimizer],[]
 