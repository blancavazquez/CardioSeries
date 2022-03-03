import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import r2_score

def r2_corr(pred,true):
    return r2_score(true, pred).mean()

def rmse_loss(pred,true):
    "Computes the root mean square error"
    criterion = torch.nn.MSELoss()
    rmse = torch.sqrt(criterion(pred,true)) 
    return rmse

def mmd_loss(outputs, targets, window_size,
             num_features,device,kernel="rbf"):

    outputs = outputs.reshape(-1,window_size*num_features,1)
    targets = targets.reshape(-1,window_size*num_features,1)

    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """

    outputs = outputs.reshape(outputs.shape[0],outputs.shape[1]*outputs.shape[2])
    targets = targets.reshape(targets.shape[0],targets.shape[1]*targets.shape[2])

    xx, yy, zz = torch.mm(outputs, outputs.t()), torch.mm(targets, targets.t()), torch.mm(outputs, targets.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
                
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