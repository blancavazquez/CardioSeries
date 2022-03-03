"""
https://github.com/jamesloyys/PyTorch-Lightning-GAN/blob/main/GAN/gan.py
THE ENCODER COMPRESSES DATA INTO A LATENT SPACE (Z). 
THE DECODER RECONSTRUCTS THE DATA GIVEN THE HIDDEN REPRESENTATION.
"""

#conv1d: https://discuss.pytorch.org/t/understanding-convolution-1d-output-and-input/30764/13

import torch
import torch.nn as nn
from math import sqrt
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
  def __init__(self,input_size):
    super().__init__()
    input_dim = 192
    output_dim = 192

    # # [N,192] =>[32,256]
    # self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim, 256),
    #                                    nn.BatchNorm1d(256),
    #                                    nn.ReLU(0.2))

    #------- Encoder -----------#
    #[N,24,8] => [N,8,32]
    self.cnn_encoder = nn.Sequential(
      nn.Conv1d(in_channels=24,out_channels=8,kernel_size=4,stride=3,padding=0,bias=False),
      nn.BatchNorm1d(8),
      nn.Dropout(p=0.25),
      nn.LeakyReLU(0.2, inplace=True))

    #------- Decoder -----------#
    # [N, 8,32] => [N, 256, 32]
    self.cnn_block_1 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=8,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
      nn.BatchNorm1d(256, momentum=0.1,  eps=0.8),
      nn.ReLU(0.2))

    # [N, 256, 32] => [N, 128, 64]
    self.cnn_block_2 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(128, momentum=0.1,  eps=0.8),
      nn.ReLU(0.2))

    # [N, 128, 64] => [N, 64, 128]
    self.cnn_block_3 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(64, momentum=0.1,  eps=0.8),
      nn.ReLU(0.2))

    # [N, 64, 128] => [N, 1, 256]
    self.cnn_block_4 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=64,out_channels=1,kernel_size=4,stride=2,padding=1,bias=False),
      nn.BatchNorm1d(1, momentum=0.1,  eps=0.8),
      nn.ReLU(0.2))

    # [N,1,16] => [32,1,256]
    self.hidden_layer2 = nn.Sequential(nn.Linear(16, 256),nn.ReLU(0.2),nn.Dropout(p=0.25),)

    # [N,1,256] => [32,1,192]
    self.hidden_layer3 = nn.Sequential(nn.Linear(256, output_dim),nn.Tanh())


  def forward(self, x_input): #[32, 24, 8]
    output = self.cnn_encoder(x_input) #[32,8,2]
    output = self.cnn_block_1(output) #[32,256,2]
    output = self.cnn_block_2(output) #[32,128,4]
    output = self.cnn_block_3(output) #[32,64,8]
    output = self.cnn_block_4(output) #[32,1,16]
    output = self.hidden_layer2(output) #[32,1,256]
    output = self.hidden_layer3(output) #[32,1,192]
    output = output.reshape(-1,x_input.shape[1],x_input.shape[2]) #[32, 24,8]
    return output

class Discriminator(nn.Module):
  def __init__(self,
               batch_size=32,
               seq_length=24,
               num_features=8):
    super().__init__()

    # [N, 1, 192] => [N, 64, 192]
    self.cnn_block_1 = nn.Sequential(
      nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
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

    # [N,192] => [N,192]
    self.hidden_layer1 = nn.Sequential(nn.Linear(192, 192), nn.Dropout(p=0.25),nn.LeakyReLU(0.2, inplace=True))

    # [N,191] => [32,1]
    self.hidden_layer2 = nn.Sequential(nn.Linear(192, 1),nn.Sigmoid())

  def weight_init(self):
    for m in self._modules:
      normal_init(self._modules[m])
    
  def forward(self, x_input): #[32,8,24]
    output = x_input.reshape(-1,1,x_input.shape[1]*x_input.shape[2]) #[32,1,192]
    output = self.cnn_block_1(output) #[32,64,192]
    output = self.cnn_block_2(output) #[32,128,192]
    output = self.cnn_block_3(output) #[32,256,192]
    output = self.cnn_block_4(output) #[32,192]
    output = self.hidden_layer1(output) #[32,192]
    output = self.hidden_layer2(output) #[32,1]
    return output


class DCGAN(pl.LightningModule):
  def __init__(self,
               lr=3e-4,
               w_decay=1e-4,
               alpha=1e-4,
               gamma=1e-4,
               target_length = 24,
               num_features = 8):

    super().__init__()
    self.generator = Generator(num_features)
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
    generated_ehr = self(x_input) #([32, 24,8]

    # Classify generated EHR (using the discriminator)
    d_output = torch.squeeze(self.discriminator(generated_ehr))#[32]

    #label
    y_true = torch.ones(x_input.shape[0], device=self.device) #[32]

    # Backprop loss
    g_loss = nn.BCELoss()(d_output, y_true)

    return g_loss

  def discriminator_step(self, x_input,x_target):
    """
    Measure discriminator's ability to classify real from generated samples
    x_input[32,24,8]  (first 24 hours)
    x_target[32,24,8]  (next 24 hours)
    """
    #-------Get real EHR-------#
    d_output = torch.squeeze(self.discriminator(x_target)) #[32]
    
    y_true = torch.ones(x_target.shape[0], device=self.device) #[32]
    
    loss_real = nn.BCELoss()(d_output, y_true)#predict_loss_real   

    #-------Get fake EHR-------#
    generated_ehr = self(x_input) #([32, 8, 24])
    
    d_output = torch.squeeze(self.discriminator(generated_ehr)) #[32]
    
    y_fake = torch.zeros(x_input.shape[0], device=self.device)#[32]
    
    loss_fake = nn.BCELoss()(d_output, y_fake) #predict_loss_fake

    d_loss = (loss_real + loss_fake)/2
    
    return d_loss

  def training_step(self, batch, batch_idx, optimizer_idx):
    target_in, target_out = batch #in [32,24,8], out [32,24,8],

    target_in = torch.tensor(target_in, dtype=torch.float32)
    target_out = torch.tensor(target_out, dtype=torch.float32)

    if optimizer_idx == 0:
      loss = self.generator_step(target_in)
      self.log("g_loss", loss)

    if optimizer_idx == 1:
      loss = self.discriminator_step(target_in,target_out)
      self.log("d_loss", loss)
    
    self.log("loss_train", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    target_in, target_out= batch #in [32,24,8], out [32,24,8]

    target_in = torch.tensor(target_in, dtype=torch.float32)
    target_out = torch.tensor(target_out, dtype=torch.float32)     

    loss = self.discriminator_step(target_in,target_out)
    self.log("loss_val",loss)
    return loss

  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr,weight_decay=self.w_decay, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr,weight_decay=self.w_decay, betas=(0.5, 0.999))
    return [g_optimizer, d_optimizer],[]
 