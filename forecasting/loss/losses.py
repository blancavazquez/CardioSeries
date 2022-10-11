import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import r2_score
from loss.dilate_loss import dilate_loss

def r2_corr(pred,true):
    return r2_score(true, pred).mean()

def ctc_loss(pred,true):
    "Computes the root mean absolute error (MAE)"
    criterion = torch.nn.CTCLoss(reduction='mean')
    ctc = criterion(pred,true,6,6)
    return ctc

def mae_loss(pred,true):
    "Computes the root mean absolute error (MAE)"
    criterion = torch.nn.L1Loss()
    mae = criterion(pred,true)
    return mae

def rmse_loss(pred,true):
    "Computes the root mean square error"
    criterion = torch.nn.MSELoss()
    rmse = torch.sqrt(criterion(pred,true)) 
    return rmse

def mse_loss(pred,true):
    "Computes the root mean square error"
    criterion = torch.nn.MSELoss()
    return criterion(pred,true)

def mmd_loss(outputs, targets, kernel="rbf"):
    
    dev = targets.device

    outputs = outputs.reshape(-1,1)
    targets = targets.reshape(-1,1)

    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(outputs, outputs.t()), torch.mm(targets, targets.t()), torch.mm(outputs, targets.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(dev),
                  torch.zeros(xx.shape).to(dev),
                  torch.zeros(xx.shape).to(dev))
                
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)

def vae_loss(pred,true,mu,logvar):
    ecm = F.mse_loss(pred,true,reduction='sum') #error_reconstruction 
    kl = -0.5 * torch.sum(1+logvar-mu**2 - logvar.exp()) #measures how much information is lost
    perdida = ecm + kl
    return perdida

def vae_dilate_loss(pred,true,mu,logvar):
    _,shape,_ = dilate_loss(pred,true,alpha=0.5,gamma=0.01,device=pred.device,missing = "True")
    kl = -0.5 * torch.sum(1+logvar-mu**2 - logvar.exp()) #measures how much information is lost
    loss = shape.item()*0.01 + kl

    print("loss:", loss,"kl:",kl.item(),"shape:",shape.item())
    print(stop)
    return loss

def vae_mmd_loss(pred,true,mu,logvar,seq_length=6,num_features=7):
    mmd = mmd_loss(pred, true, seq_length,num_features,pred.device)
    kl = -0.5 * torch.sum(1+logvar-mu**2 - logvar.exp()) #measures how much information is lost
    loss = (mmd + kl)/2
    return loss

def DiceBCELoss(self, ypred, yreal, smooth=1):
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