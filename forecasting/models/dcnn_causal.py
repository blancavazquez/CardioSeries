#Code based on https://github.com/kristpapadopoulos/seriesnet/blob/master/seriesnet.py

import torch
import numpy as np
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from loss.dilate_loss import dilate_loss
from loss.losses import r2_corr,rmse_loss,mmd_loss,mae_loss, ctc_loss
from utils import plotting_predictions,dwprobability,saving_logs_validation,saving_logs_training
from torchmetrics import R2Score

##############################################################################
##############################################################################

def weight_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.trunc_normal_(m.weight, 0.0, 0.05)

class rnn(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, batch_size,num_features,output_rnn):
        super(rnn, self).__init__()  

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_rnn = output_rnn
        self.num_features = num_features

        self.dense = nn.Sequential(nn.Linear(5, output_rnn),nn.Mish(),nn.Dropout(p=0.8))
        self.rnn = nn.LSTM(input_size=self.num_features, hidden_size=50, num_layers=1,
                            batch_first=True,bidirectional=True)

    def forward(self, x_input, cond): #input: ([32, 6, 7]) cond: ([32, 5]) 
        
        # [32,5] => [2,16,5]
        cond = cond.reshape(2,-1,cond.shape[1])

        #[1,32,5] => [2,16,100]
        h0 = self.dense(cond)
        h0 = h0.reshape(2,x_input.shape[0],-1) #[2,32,50]    
        output, _ = self.rnn(x_input, (h0,h0)) # para LSTM

        return output

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,nb_filter,filter_length,dilation,output_rnn,device,input_seq_length,num_features):
        self.__padding = ((filter_length - 1) * dilation)
        super(CausalConv1d, self).__init__(7,out_channels = nb_filter,kernel_size=filter_length,
            stride=1,padding=self.__padding,dilation=dilation,groups=1,bias=False,device=device)

        self.rnn = rnn(input_size = input_seq_length, hidden_size = 128, num_layers = 1, batch_size = 32,
                        num_features= num_features, output_rnn=output_rnn)
        self.linear = nn.Linear(output_rnn, num_features)

    def forward(self, x_input,cond): 
        """ 
        Parameters:
        x_input = [batch_size,input_seqlength,num_features], 
        cond =[batch_size,num_conditions]

        Return:
        conv1d_out = [batch_size,32,input_seqlength]
        """

        if cond.shape[1] > 0: # Condicionando la red
            
            #[batch_size,input_seqlength,num_features], [batch_size, num_cond] => [batch_size,input_seqlength,100]
            out_rnn = self.rnn(x_input, cond)
            out_rnn = self.linear(out_rnn) #[32,6,7]

            #[batch_size,num_features,input_seqlength] => [batch_size,input_seqlength,num_features]
            out_rnn = out_rnn.permute(0,2,1)
        
        else: #no condicionada
            out_rnn = x_input.permute(0,2,1)

        #[batch_size,input_seqlength,num_features] => [batch_size,32,input_seqlength]
        conv1d_out = super(CausalConv1d, self).forward(out_rnn)

        #[batch_size,32,input_seqlength] => [batch_size,32,input_seqlength]
        conv1d_out = conv1d_out[:, :, :-self.__padding] if self.__padding != 0 else conv1d_out
        return conv1d_out

class DC_CNN_Block(torch.nn.Module):
    def __init__(self,nb_filter, filter_length, dilation, output_rnn, num_features,input_seq_length):
        super(DC_CNN_Block, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nb_filter = nb_filter
        self.filter_length= filter_length
        self.dilation = dilation
        self.output_rnn = output_rnn
        self.num_features = num_features
        self.input_seq_length = input_seq_length
        padding = 0 if self.input_seq_length == 6 else 1
        stride = 6 if self.input_seq_length == 6 else 4
        

        self.causal = CausalConv1d(nb_filter = self.nb_filter,filter_length = self.filter_length,
                                dilation= self.dilation,output_rnn = self.output_rnn ,device=device,
                                input_seq_length=self.input_seq_length,num_features=self.num_features)
        self.skip_out = nn.Conv1d(input_seq_length,out_channels=self.num_features,kernel_size=1,stride=stride,padding=padding,bias=False,device=device)
        self.network_in = nn.Conv1d(input_seq_length,out_channels=self.num_features,kernel_size=1,stride=stride,padding=padding,bias=False,device=device)
        
        # --- Initialize weights---
        self.causal.apply(weight_init) 
        self.skip_out.apply(weight_init)
        self.network_in.apply(weight_init)

    def forward(self, x_input,cond):
        dev = x_input.device

        residual = x_input #[32,6,7]

        #[32, 6, 7]) cond: torch.Size([32, 5] => [32, 32, 6]
        layer_out = self.causal(x_input,cond).to(dev)

        #[32, 32, 6] => [32, 6, 32]
        layer_out = layer_out.permute(0,2,1).to(dev)
        layer_out = F.selu(layer_out)

        #[32, 6, 32] => [32, 7, 6]
        skip_out = self.skip_out(layer_out)
        skip_out = skip_out.permute(0,2,1).to(dev) #[32, 6, 7]

        #[32, 6, 32] => [32, 7, 6]
        network_in = self.network_in(layer_out)
        network_in = network_in.permute(0,2,1).to(dev) #[32, 6, 7]
        
        #[32, 6, 7] + [32, 6, 7] => [32, 6, 7]
        network_out = residual + network_in
        return network_out, skip_out


class CausalModel(pl.LightningModule):
    def __init__(self, w_decay,dropout,alpha,gamma,input_seq_length,output_seq_length,
                 output_rnn,num_features,batch_size,lr,num_conditions, path,feature_list,
                 net,in_channels,out_channels,kernel_size,stride,dilation,groups,bias):

        super(CausalModel,self).__init__()
        self.w_decay = w_decay
        self.dropout = dropout
        self.alpha = alpha
        self.gamma = gamma
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.output_rnn = output_rnn
        self.num_features = num_features
        self.batch_size = batch_size
        self.lr = lr
        self.num_conditions = num_conditions
        self.path = path
        self.feature_list = feature_list
        self.net = net
        self.save_hyperparameters()
    
        self.loss = 0
        self.val_loss = 0

        self.block_1 = DC_CNN_Block(nb_filter=32, filter_length=2, dilation=16, output_rnn = self.output_rnn, num_features=self.num_features,input_seq_length=self.input_seq_length)
        self.block_2 = DC_CNN_Block(nb_filter=32, filter_length=2, dilation=32, output_rnn = self.output_rnn, num_features=self.num_features,input_seq_length=self.input_seq_length)
        self.block_3 = DC_CNN_Block(nb_filter=32, filter_length=2, dilation=64, output_rnn = self.output_rnn, num_features=self.num_features,input_seq_length=self.input_seq_length)
        
        stride = 1 if self.input_seq_length == 6 else 4
        padding = 0 if self.input_seq_length == 6 else 1
        self.l21 = nn.Conv1d(self.num_features,out_channels=self.num_features,kernel_size=1,stride=stride,padding=padding,bias=False)

        # Initialize weights 
        self.block_1.apply(weight_init)
        self.block_2.apply(weight_init)
        self.block_3.apply(weight_init)
        self.l21.apply(weight_init)

    def forward(self,x_input,cond): 
        """
        input sequence: [batch_size, input_seq_length,num_features]
        condition: [batch_size,num_conditions]
        """
        dev = x_input.device #[32,6,7]

        l1a, l1b = self.block_1(x_input,cond)
        l2a, l2b = self.block_2(l1a,cond)

        l3b = nn.Dropout(0.8)(l2b)
        l4a, l4b = self.block_3(l2a,cond)
        l4b = nn.Dropout(0.8)(l4b)
        l5 = l1b + l2b + l3b + l4b

        l6 = F.relu(l5)
        l6 = l6.permute(0,2,1).to(dev)
        l21 = self.l21(l6)
        l21 = l21.permute(0,2,1).to(dev)
        return l21

    def training_step(self, batch, batch_idx):
        target_in, target_out,condition, mask = batch
        condition = condition[:,:self.num_conditions] #settings to number of conditions
        target_in = torch.tensor(target_in, dtype=torch.float32).to(target_in.device)#[32, 6, 7]
        target_out = torch.tensor(target_out, dtype=torch.float32).to(target_out.device)#[32, 6, 7]       
        
        #[32, 6, 7], [32,5] => [32, 6, 7]
        ypred = self(target_in,condition)

        mae = mae_loss(ypred,target_out)
        mmd = mmd_loss(ypred,target_out)
        rmse = rmse_loss(ypred,target_out)
        dilate,loss_shape,loss_temporal = dilate_loss(ypred,target_out,batch_size=self.batch_size,
                                                      seq_length = self.output_seq_length,num_features=self.num_features,
                                                      alpha=self.alpha,gamma=self.gamma,device=self.device,
                                                      mask = "False")
        self.log("loss",rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss":rmse,"past":target_in,"ytrue":target_out,"ypred":ypred,"conditions":condition, "mask":mask}

    def training_epoch_end(self, training_step_outputs):
        loss_training = torch.flatten(torch.stack([x['loss'] for x in training_step_outputs]))
        loss_training = loss_training.view(-1).detach().cpu().numpy().flatten()
        saving_logs_training(loss_training)

        # Convert from list to tensor
        past = torch.flatten(torch.stack([x['past'] for x in training_step_outputs]))
        ytrue = torch.flatten(torch.stack([x['ytrue'] for x in training_step_outputs]))
        ypred = torch.flatten(torch.stack([x['ypred'] for x in training_step_outputs]))
        conditions = torch.flatten(torch.stack([x['conditions'] for x in training_step_outputs]))
        mask = torch.flatten(torch.stack([x['mask'] for x in training_step_outputs]))
        past = past.view(-1).detach().cpu().numpy().reshape(-1,1)
        ytrue = ytrue.view(-1).detach().cpu().numpy().reshape(-1,1)
        ypred = ypred.view(-1).detach().cpu().numpy().reshape(-1,1)
        conditions = conditions.view(-1).detach().cpu().numpy().reshape(-1,1)
        mask = mask.view(-1).detach().cpu().numpy().reshape(-1,1)
        plotting_predictions(past,ytrue,ypred,mask,self.input_seq_length,self.output_seq_length,
                             self.num_features,self.num_conditions,conditions,self.path,"dcnn",1,"train",self.current_epoch)

    def validation_step(self, batch, batch_idx):
        target_in, target_out,condition, mask = batch
        condition = condition[:,:self.num_conditions] #settings to number of conditions
        target_in = torch.tensor(target_in, dtype=torch.float32).to(target_in.device)#[32, 6, 7]
        target_out = torch.tensor(target_out, dtype=torch.float32).to(target_out.device)#[32, 6, 7]
        
        #[32, 6, 7], [32,5] => [32, 6, 7] 
        ypred = self(target_in,condition)

        #------- Computing loss (on real data, using masking)-------#
        ypred_mask = torch.masked_select(ypred, mask)
        target_mask = torch.masked_select(target_out, mask)
        #-----------------------------------------------------------#

        mae = mae_loss(ypred_mask,target_mask)
        mmd = mmd_loss(ypred_mask,target_mask)
        rmse = rmse_loss(ypred_mask,target_mask)
        dilate,loss_shape,loss_temporal = dilate_loss(ypred_mask,target_mask,batch_size=self.batch_size,
                                                      seq_length = self.output_seq_length,num_features=self.num_features,
                                                      alpha=self.alpha,gamma=self.gamma,device=self.device,
                                                      mask = "True")
        r2score = R2Score().to(target_in.device)
        score_R2 = r2score(ypred_mask.reshape(-1),target_mask.reshape(-1))
        self.log("val_loss",rmse)
        return {"val_loss":rmse,
                "rmse":rmse,"dilate":dilate,"shape":loss_shape,"temporal":loss_temporal,"r2_corr":score_R2,
                "past":target_in,"ytrue":target_out,"ypred":ypred,"conditions":condition, "mask":mask}

    def validation_epoch_end(self, validation_step_outputs):
        loss_val = torch.flatten(torch.stack([x['val_loss'] for x in validation_step_outputs]))
        loss_val = loss_val.view(-1).detach().cpu().numpy().flatten()

        rmse_val = torch.flatten(torch.stack([x['rmse'] for x in validation_step_outputs]))
        dilate_val = torch.flatten(torch.stack([x['dilate'] for x in validation_step_outputs]))
        shape_val = torch.flatten(torch.stack([x['shape'] for x in validation_step_outputs]))
        temporal_val = torch.flatten(torch.stack([x['temporal'] for x in validation_step_outputs]))
        r2_val = torch.flatten(torch.stack([x['r2_corr'] for x in validation_step_outputs]))

        ##Extracción de solo valores
        rmse_val = rmse_val.view(-1).detach().cpu().numpy().flatten()
        dilate_val = dilate_val.view(-1).detach().cpu().numpy().flatten()
        shape_val = shape_val.view(-1).detach().cpu().numpy().flatten()
        temporal_val = temporal_val.view(-1).detach().cpu().numpy().flatten()
        r2_val = r2_val.view(-1).detach().cpu().numpy().flatten()

        saving_logs_validation(loss_val,rmse_val,dilate_val,shape_val,temporal_val,r2_val)

        # Convert from list to tensor
        past_val = torch.flatten(torch.stack([x['past'] for x in validation_step_outputs]))
        ytrue_val = torch.flatten(torch.stack([x['ytrue'] for x in validation_step_outputs]))
        ypred_val = torch.flatten(torch.stack([x['ypred'] for x in validation_step_outputs]))
        conditions_val = torch.flatten(torch.stack([x['conditions'] for x in validation_step_outputs]))
        mask_val = torch.flatten(torch.stack([x['mask'] for x in validation_step_outputs]))

        past_val = past_val.view(-1).detach().cpu().numpy().reshape(-1,1)
        ytrue_val = ytrue_val.view(-1).detach().cpu().numpy().reshape(-1,1)
        ypred_val = ypred_val.view(-1).detach().cpu().numpy().reshape(-1,1)
        conditions_val = conditions_val.view(-1).detach().cpu().numpy().reshape(-1,1)
        mask_val = mask_val.view(-1).detach().cpu().numpy().reshape(-1,1)

        plotting_predictions(past_val,ytrue_val,ypred_val,mask_val,self.input_seq_length,self.output_seq_length,
                             self.num_features,self.num_conditions,conditions_val,self.path,"dcnn",1,"val",self.current_epoch)
        dwprobability(ytrue_val,ypred_val,self.output_seq_length,self.num_features,self.path,self.net,self.current_epoch,"val")

    def test_step(self, batch, batch_idx): #Se añadió para la parte de Kfold
        target_in, target_out,condition, mask = batch
        condition = condition[:,:self.num_conditions] #settings to number of conditions
        target_in = torch.tensor(target_in, dtype=torch.float32).to(target_in.device)#[32, 6, 7]
        target_out = torch.tensor(target_out, dtype=torch.float32).to(target_out.device)#[32, 6, 7]
        
        #[32, 6, 7], [32,5] => [32, 6, 7] 
        ypred = self(target_in,condition)

        #------- Computing loss (on real data, using masking)-------#
        ypred_mask = torch.masked_select(ypred, mask)
        target_mask = torch.masked_select(target_out, mask)
        #-----------------------------------------------------------#

        mae = mae_loss(ypred_mask,target_mask)
        mmd = mmd_loss(ypred_mask,target_mask)
        rmse = rmse_loss(ypred_mask,target_mask)
        dilate,loss_shape,loss_temporal = dilate_loss(ypred_mask,target_mask,batch_size=self.batch_size,
                                                       seq_length = self.output_seq_length,num_features=self.num_features,
                                                       alpha=self.alpha,gamma=self.gamma,device=self.device,
                                                       mask = "True")
        self.log("test_loss",rmse)
        return {"test_loss":rmse}

    def configure_optimizers(self):
        """
        This means that model.base‘s parameters will use the default learning rate of 5e-5, 
        model.classifier‘s parameters will use a learning rate of 1e-3, 
        and a weight_decay and betas will be used for all parameters
        """
        optimizer = torch.optim.RMSprop([ #AdamW
                {'params': self.block_1.parameters(),'lr':5e-4},
                {'params': self.block_2.parameters(), 'lr': 5e-4},
                {'params': self.block_3.parameters(), 'lr':5e-3},
                {'params':self.l21.parameters(),'lr':5e-3}],
                lr=5e-5, weight_decay=1e-8) #,betas=(0.9, 0.999)) #añadir betas a AdamW

        # T_0 (int) – Number of iterations for the first restart.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,T_mult=1, eta_min=0.001, last_epoch=-1)
        lrs = []
        for epoch in range(100):
            scheduler.step(epoch)
            #lrs.append(scheduler.get_last_lr())
            lrs.append(optimizer.param_groups[0]["lr"])

        lrs = np.array(lrs)
        plt.plot(lrs, label="LR")
        plt.legend()
        plt.savefig(self.path+'plots/lr_cosine.png')
        return {"optimizer": optimizer,"lr_scheduler": scheduler,"monitor": "val_loss"}
