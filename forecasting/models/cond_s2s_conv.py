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

        # [N, 1, 8] => [N, 64, 4]
        self.cnn_block_1 = nn.Sequential(
          nn.Conv1d(in_channels=1,out_channels=64,kernel_size=2,stride=2,padding=0,bias=False),
          nn.BatchNorm1d(64),
          nn.Dropout(p=0.25),
          nn.LeakyReLU(0.2, inplace=True),
          )

        # [N, 64, 4] => [N,128,4]
        self.cnn_block_2 = nn.Sequential(
          nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False),
          nn.BatchNorm1d(128),
          nn.Dropout(p=0.25),
          nn.LeakyReLU(0.2, inplace=True))

        # [N, 128, 4] => [N,256,4]
        self.cnn_block_3 = nn.Sequential(
          nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
          nn.BatchNorm1d(256),
          nn.Dropout(p=0.25),
          nn.LeakyReLU(0.2, inplace=True))

        # [N, 256, 4] => [N,1,4]
        self.cnn_block_4 = nn.Sequential(
          nn.Conv1d(in_channels=256,out_channels=8,kernel_size=3,stride=1,padding=1,bias=False),
          nn.Dropout(p=0.25))

        self.gru = nn.GRU(input_size=4, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden): #input: ([32, 1, 8]) hidden: ([1, 32, 128])
        output = self.cnn_block_1(input) #[32,64,4]
        output = self.cnn_block_2(output) #[32,128,4]
        output = self.cnn_block_3(output) #[32,256,4]
        output = self.cnn_block_4(output) #[32,8,4]
        output, hidden = self.gru(output, hidden)
        return output, hidden #out[32,1,128], h[1,32,128]
    
    def init_hidden(self,device):
        output = torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device="cuda:0")
        return output
    
class decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers,fc_units, output_size):
        super(decoder, self).__init__()

        # [N, 10,1] => [N, 256, 4]
        self.cnn_block_1 = nn.Sequential(
          nn.ConvTranspose1d(in_channels=10,out_channels=256,kernel_size=4,stride=2,padding=0,bias=False),
          nn.BatchNorm1d(256, momentum=0.1,  eps=0.8),
          nn.ReLU(0.2))

        # [N, 256, 4] => [N, 128, 8]
        self.cnn_block_2 = nn.Sequential(
          nn.ConvTranspose1d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
          nn.BatchNorm1d(128, momentum=0.1,  eps=0.8),
          nn.ReLU(0.2))

        # [N, 128, 8] => [N, 64, 16]
        self.cnn_block_3 = nn.Sequential(
          nn.ConvTranspose1d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
          nn.BatchNorm1d(64, momentum=0.1,  eps=0.8),
          nn.ReLU(0.2))

        # [N, 64, 16] => [N, 1, 32]
        self.cnn_block_4 = nn.Sequential(
          nn.ConvTranspose1d(in_channels=64,out_channels=1,kernel_size=4,stride=2,padding=1,bias=False),
          nn.BatchNorm1d(1, momentum=0.1,  eps=0.8),
          nn.ReLU(0.2))

        # [N,1,32] => [N,1,10]
        self.hidden_layer = nn.Sequential(nn.Linear(32, 10),nn.ReLU(0.2),nn.Dropout(p=0.25),)

        # [N,1,10] => out[32,1,128], h[1,32,128]
        self.gru = nn.GRU(input_size=10, hidden_size=hidden_size,num_layers=num_grulstm_layers,batch_first=True)
        
        # [N,1,128] => [32,1,16]
        self.fc = nn.Linear(hidden_size, fc_units)
        
        # [N,1,16] => [N,1,8]
        self.out = nn.Linear(fc_units, output_size)         
        
    def forward(self, x_input, hidden,cond): #x[32,1,8], h[1,32,128], c[32,2]
        x_input_np = x_input.reshape(-1,x_input.shape[1]*x_input.shape[2]) #[32, 8]
        cond = cond.long() #[32,2] 
        output = torch.cat([x_input_np,cond],1) #[32,10]
        output = output.reshape(-1,1,10) #[32,1,10]
        output = output.permute((0, 2, 1)) #[32, 10, 1]
        output = self.cnn_block_1(output) #[32,256,4]
        output = self.cnn_block_2(output) #[32,128,8]
        output = self.cnn_block_3(output) #[32,64,16]
        output = self.cnn_block_4(output) #[32,1,32]
        output = self.hidden_layer(output) #[32,1,10]
        output, hidden = self.gru(output, hidden)
        output = F.relu( self.fc(output))
        output = self.out(output)
        return output, hidden #out[32,1,8], h[1,32,128]
    
class CondS2S_conv(pl.LightningModule):
    def __init__(self, encoder, decoder,w_decay,
                 dropout,alpha,gamma,target_length,num_features,lr):
        super(CondS2S_conv, self).__init__()
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
        encoder_hidden = self.encoder.init_hidden(x_input.device) #[1,32,128]

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x_input[:,ei:ei+1,:],encoder_hidden) #out[32, 8, 128], h[1, 32, 128]

        decoder_input = x_input[:,-1,:].unsqueeze(1)# [32,1,8] first decoder input = last element of input sequence
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
        #loss = rmse_loss(y_pred,target_out)
        loss,loss_shape,loss_temporal = dilate_loss(y_pred,target_out,alpha=self.alpha,gamma=self.gamma,device=self.device)
        self.log("loss_train", loss)
        return loss

    def validation_step(self,batch,batch_idx):
        target_in, target_out, conditions = batch
        y_pred = self(target_in,conditions)#[32, 24, 8]
        #loss = rmse_loss(y_pred,target_out)
        loss,loss_shape,loss_temporal = dilate_loss(y_pred,target_out,alpha=self.alpha,gamma=self.gamma,device=self.device)
        self.log("loss_val",loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.w_decay)        
        return {"optimizer": optimizer,"monitor": "loss",}      