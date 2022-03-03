import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import pytorch_lightning as pl
from loss.dilate_loss import dilate_loss
from loss.losses import r2_corr,rmse_loss,mmd_loss

class encoder(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size):
        super(encoder, self).__init__()  

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                          num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]      
        output, hidden = self.gru(input, hidden)      
        return output, hidden
    
    def init_hidden(self,device):
        #[num_layers*num_directions,batch,hidden_size]   
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device="cuda:0")
    
class decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers,fc_units, output_size):
        super(decoder, self).__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                          num_layers=num_grulstm_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size, fc_units)
        
        self.out = nn.Linear(fc_units, output_size)         
        
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden) 
        output = F.relu( self.fc(output) )
        output = self.out(output)      
        return output, hidden
    
class encoder_decoder(pl.LightningModule):
    def __init__(self, encoder, decoder,w_decay,
                 dropout,alpha,gamma,target_length,num_features,lr):
        super(encoder_decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.w_decay = w_decay
        self.dropout = dropout
        self.alpha = alpha
        self.gamma = gamma
        self.target_length = target_length
        self.num_features = num_features
        self.lr = lr
        self.save_hyperparameters()
                
    def forward(self, x):

        input_length  = x.shape[1] #192
        encoder_hidden = self.encoder.init_hidden(x.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:,ei:ei+1,:]  , encoder_hidden)
            
        decoder_input = x[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden
        
        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]).to(x.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:,di:di+1,:] = decoder_output
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, targets  = batch #[32, 24, 1, 8], [32, 24, 1, 8])

        inputs = torch.tensor(inputs, dtype=torch.float32).to(inputs.device)#[32, 24, 8]
        y_real = torch.tensor(targets, dtype=torch.float32).to(inputs.device)#[32, 24, 8]
        
        y_pred = self(inputs)#[32, 24, 8]

        #loss = mmd_loss(y_pred,y_real,inputs.shape[1],inputs.shape[3],device=inputs.device)

        loss = rmse_loss(y_pred,y_real)

        # loss, loss_shape, loss_temporal = dilate_loss(y_pred, y_real, alpha=self.alpha, 
        #                                               gamma=self.gamma,device=inputs.device)
        self.log("loss_train", loss)
        return loss


    def validation_step(self,batch,batch_idx):
        inputs,targets = batch

        inputs = torch.tensor(inputs, dtype=torch.float32).to(inputs.device)#[32, 24, 8]
        y_real = torch.tensor(targets, dtype=torch.float32).to(inputs.device)#[32, 24, 8]
        
        y_pred = self(inputs)

        #loss = mmd_loss(y_pred,y_real,inputs.shape[1],inputs.shape[3],device=inputs.device)
        
        loss = rmse_loss(y_pred,y_real)

        # loss,loss_shape,loss_temporal = dilate_loss(y_pred,y_real,alpha=self.alpha,
        #                                             gamma=self.gamma,device=inputs.device)
        self.log("loss_val",loss)
        return loss

    def test_step(self,batch,batch_idx): #recorre TODO el val_set por batch
        inputs,targets = batch

        inputs = torch.tensor(inputs, dtype=torch.float32).to(inputs.device)#[32, 24, 8]
        y_real = torch.tensor(targets, dtype=torch.float32).to(inputs.device)#[32, 24, 8]

        y_pred = self(inputs)

        loss,loss_shape,loss_temporal = dilate_loss(y_pred,y_real,alpha=self.alpha,
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