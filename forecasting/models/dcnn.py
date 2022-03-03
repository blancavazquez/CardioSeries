import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import pytorch_lightning as pl
from loss.dilate_loss import dilate_loss
from loss.losses import r2_corr,rmse_loss,mmd_loss


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
      torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm1d') != -1:
      torch.nn.init.normal(m.weight, 1.0, 0.02)
      torch.nn.init.constant(m.bias, 0.0)

class DC_CNN_Block(nn.Module):
  def __init__(self):
    super().__init__()

    # [N,8,24] => [N,32,23]
    self.DC_CNN_Block_1 = nn.Sequential(
              nn.Conv1d(in_channels=8,out_channels=32,kernel_size=2,dilation=1,padding_mode='circular', bias=False),
              nn.BatchNorm1d(32),
              nn.Dropout(p=0.25),
              nn.SELU())

    #[N,32,23] => [N,32,23]
    self.skip_out = nn.Sequential(
              nn.Conv1d(in_channels=32,out_channels=32,kernel_size=2,dilation=1,bias=False),
              nn.BatchNorm1d(32),
              nn.Dropout(p=0.25))

    #[N,32,22] => [N,1,22]
    self.network_in = nn.Sequential(
              nn.Conv1d(in_channels=32,out_channels=1,kernel_size=2,dilation=1,bias=False),
              nn.BatchNorm1d(1),
              nn.Dropout(p=0.25))

    
  def forward(self, x_input): #x_input=[32,24,8]
    print("dentro de forward:", x_input.shape)
    x_input = x_input.permute((0,2,1)) #[32,8,24]
    output = self.DC_CNN_Block_1(x_input) #[32,32,23]
    print("DC_CNN_Block_1:", output.shape)

    skip = self.skip_out(output) #[32,32,23]
    print("skip_out:", output.shape)

    net = self.network_in(output) #[32,1,22]
    print("network_in:", net.shape)

    x_input_np = x_input.reshape(-1,x_input.shape[1]*x_input.shape[2]) #[32,192]
    net_np = net.reshape(-1,net.shape[1]*net.shape[2]) #[32,22]
    out = torch.cat([x_input_np,net_np],1) #[32,214]
    print("out:", out.shape)

    print("----------------------")
    print(stop)
    return output

class DC_CNN_Model(pl.LightningModule):
    def __init__(self, target_length,num_features,w_decay,
                 dropout,alpha,gamma,lr):
        super(DC_CNN_Model,self).__init__()
        self.DC_CNN_Block = DC_CNN_Block()
        self.target_length = target_length
        self.num_features = num_features
        self.w_decay = w_decay
        self.dropout = dropout
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.save_hyperparameters()

        # Initialize weights
        self.DC_CNN_Block.apply(weight_init)

    def forward(self, x_input): #x[32,24,8]
        out = self.DC_CNN_Block(x_input)
        print("out:", out.shape)

        print(stop)
        return out

    def training_step(self, batch, batch_idx):
        print("training_step")

        inputs, targets  = batch #[32, 24, 1, 8], [32, 24, 1, 8])

        inputs = torch.tensor(inputs, dtype=torch.float32).to(inputs.device)#[32, 24, 8]
        y_real = torch.tensor(targets, dtype=torch.float32).to(inputs.device)#[32, 24, 8]
        
        y_pred = self(inputs)#[32, 24, 8]

        loss = rmse_loss(y_pred,y_real)

        self.log("loss_train", loss)
        return loss

    def configure_optimizers(self):
        print("configure_optimizers:", self)
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.w_decay)        
        return {"optimizer": optimizer,"monitor": "loss",}      