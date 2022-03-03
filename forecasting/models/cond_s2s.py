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

    def forward(self, input, hidden): #input: ([32, 1, 8]) hidden: ([1, 32, 128])   
        output, hidden = self.gru(input, hidden)
        return output, hidden
    
    def init_hidden(self,device): 
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device="cuda:0")
    
class decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers,fc_units, output_size):
        super(decoder, self).__init__()

        self.hidden_layer = nn.Sequential(nn.Linear(10, 8),
                                           nn.ReLU(0.2),
                                           nn.Dropout(p=0.25),)

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                          num_layers=num_grulstm_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size, fc_units)
        
        self.out = nn.Linear(fc_units, output_size)         
        
    def forward(self, x_input, hidden,cond):
        x_input_np = x_input.reshape(-1,x_input.shape[1]*x_input.shape[2]) #[32, 8]
        cond = cond.long() #[32,2] 
        output = torch.cat([x_input_np,cond],1) #[32,10]
        output = output.reshape(-1,1,10) #[32,1,10]

        x_input = self.hidden_layer(output) #[32,1,8]
        output, hidden = self.gru(x_input, hidden)
        output = F.relu( self.fc(output) )
        output = self.out(output)
 
        return output, hidden
    
class CondS2S(pl.LightningModule):
    def __init__(self, encoder, decoder,w_decay,
                 dropout,alpha,gamma,target_length,num_features,lr):
        super(CondS2S, self).__init__()
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
                
    def forward(self, x_input,cond):
        input_length  = x_input.shape[1] #192
        encoder_hidden = self.encoder.init_hidden(x_input.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x_input[:,ei:ei+1,:]  , encoder_hidden) #[32, 1, 128], [1, 32, 128]

        decoder_input = x_input[:,-1,:].unsqueeze(1)# [32,1,8] first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden #[1, 32, 128]
        
        outputs = torch.zeros([x_input.shape[0], self.target_length, x_input.shape[2]]).to(x_input.device) #[32, 24, 8]

        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden,cond) #[32,1,8], [1,32,128]
            decoder_input = decoder_output #[32,1,8]
            outputs[:,di:di+1,:] = decoder_output #[32,1,8]
        return outputs

    def training_step(self, batch, batch_idx):
        target_in, target_out, conditions = batch
        y_pred = self(target_in,conditions)#[32, 24, 8]
        loss = rmse_loss(y_pred,target_out)
        #loss,loss_shape,loss_temporal = dilate_loss(y_pred,target_out,alpha=self.alpha,gamma=self.gamma,device=self.device)
        self.log("loss_train", loss)
        return loss

    def validation_step(self,batch,batch_idx):
        target_in, target_out, conditions = batch
        y_pred = self(target_in,conditions)#[32, 24, 8]
        loss = rmse_loss(y_pred,target_out)
        #loss,loss_shape,loss_temporal = dilate_loss(y_pred,target_out,alpha=self.alpha,gamma=self.gamma,device=self.device)
        self.log("loss_val",loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.w_decay)        
        return {"optimizer": optimizer,"monitor": "loss",}      