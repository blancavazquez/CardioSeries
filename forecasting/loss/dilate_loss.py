import torch
from . import soft_dtw
from . import path_soft_dtw 

def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)

	outputs=outputs.reshape(-1,24*8,1)
	targets=targets.reshape(-1,24*8,1)

	#N_ouputs = seq_length * num_features
	batch_size, N_output = outputs.shape[0:2] #batch_size: 2 N_output: 192

	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply

	D = torch.zeros((batch_size, N_output,N_output )).to(device) #original!!!! (2,192,192)

	for k in range(batch_size):
		#Compute the distances
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1)) #torch.Size([192, 192])

		D[k:k+1,:,:] = Dk 

	loss_shape = softdtw_batch(D,gamma)	#SoftDTWBatchBackward
	#print("loss_shape:", loss_shape)

	path_dtw = path_soft_dtw.PathDTWBatch.apply

	path = path_dtw(D,gamma) #PathDTWBatchBackward #torch.Size([192, 192])

	#Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device) #torch.Size([192, 192])

	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) #DivBackward0 #torch.Size([])
	#print("loss_temporal:", loss_temporal)
	 
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal

	return loss, loss_shape, loss_temporal
