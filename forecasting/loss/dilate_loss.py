import torch
import numpy as np
from . import soft_dtw
from . import path_soft_dtw 

def pad_arr(arr, batch_size,seq_length,num_features):
    """
    Pad top of array when there is not enough history
    """
    expected_size = batch_size*seq_length*num_features
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode='edge').reshape(-1,seq_length*num_features,1)
    arr = torch.tensor(arr)
    return arr

def dilate_loss(outputs, targets, batch_size,seq_length,num_features,alpha, gamma, device,mask):
	dev = targets.device

	if mask == "True":
		outputs = pad_arr(outputs.reshape(-1).detach().cpu().numpy().reshape(-1,1),batch_size,seq_length,num_features).to(dev) #[32,42,1]
		targets = pad_arr(targets.reshape(-1).detach().cpu().numpy().reshape(-1,1),batch_size,seq_length,num_features).to(dev)

	else:#Sin masking
		outputs=outputs.reshape(-1,outputs.shape[1]*outputs.shape[2],1) #[32,42,1]
		targets=targets.reshape(-1,targets.shape[1]*targets.shape[2],1) ##[32,42,1]

	#N_ouputs = seq_length * num_features
	batch_size, N_output = outputs.shape[0:2] #batch_size: 2 N_output: 192

	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output ),requires_grad=True).to(dev) #original!!!! (32,42,42)

	for k in range(batch_size):
		#Compute the distances
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1)) #torch.Size([42, 42])
		D[k:k+1,:,:] = Dk

	loss_shape = softdtw_batch(D,gamma)	#SoftDTWBatchBackward
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma) #PathDTWBatchBackward #torch.Size([192, 192])

	#Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(dev) #torch.Size([192, 192])
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) #DivBackward0 #torch.Size([])	 
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss, loss_shape, loss_temporal
