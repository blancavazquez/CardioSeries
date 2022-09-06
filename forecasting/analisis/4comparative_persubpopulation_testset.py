import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

#------ Settings --------
title = "Rendimientos de los modelos en el conjunto de prueba \n (tamaño de la predicción = 24 horas, con condicionamiento)"
filename = "boxplot_prueba_24horas_shape"
path = "causal_in6_out6_c5_shape"
#-------------------------

#Opening loss_on_valset
acs_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs/rmse.txt', sep=" ")
acs_dilate = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs/dilate.txt', sep=" ")
acs_shape = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs/shape.txt', sep=" ")
acs_temporal = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs/temporal.txt', sep=" ")
acs_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs/r2.txt', sep=" ")

stemi_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi/rmse.txt', sep=" ")
stemi_dilate = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi/dilate.txt', sep=" ")
stemi_shape = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi/shape.txt', sep=" ")
stemi_temporal = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi/temporal.txt', sep=" ")
stemi_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi/r2.txt', sep=" ")

nstemi_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi/rmse.txt', sep=" ")
nstemi_dilate = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi/dilate.txt', sep=" ")
nstemi_shape = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi/shape.txt', sep=" ")
nstemi_temporal = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi/temporal.txt', sep=" ")
nstemi_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi/r2.txt', sep=" ")

print("acs_rmse:", acs_rmse.shape, "acs_dilate:", acs_dilate.shape, "acs_shape:", acs_shape.shape, "acs_temporal:", acs_temporal.shape,"acs_r2:", acs_r2.shape)
print("stemi_rmse:", stemi_rmse.shape, "stemi_dilate:", stemi_dilate.shape, "stemi_shape:", stemi_shape.shape, "stemi_temporal:", stemi_temporal.shape,"stemi_r2:", stemi_r2.shape)
print("nstemi_rmse:", nstemi_rmse.shape, "nstemi_dilate:", nstemi_dilate.shape, "nstemi_shape:", nstemi_shape.shape, "nstemi_temporal:", nstemi_temporal.shape,"nstemi_r2:", nstemi_r2.shape)


data_acs = pd.DataFrame({'acs_rmse':acs_rmse.to_numpy().flatten(),
                        'acs_dilate':acs_dilate.to_numpy().flatten(),
                        'acs_shape':acs_shape.to_numpy().flatten(),
                        'acs_temporal':acs_temporal.to_numpy().flatten(),
                        'acs_r2':acs_r2.to_numpy().flatten(),})

data_stemi = pd.DataFrame({'stemi_rmse':stemi_rmse.to_numpy().flatten(),
                        'stemi_dilate':stemi_dilate.to_numpy().flatten(),
                        'stemi_shape':stemi_shape.to_numpy().flatten(),
                        'stemi_temporal':stemi_temporal.to_numpy().flatten(),
                        'stemi_r2':stemi_r2.to_numpy().flatten(),})

data_nstemi = pd.DataFrame({'nstemi_rmse':nstemi_rmse.to_numpy().flatten(),
                        'nstemi_dilate':nstemi_dilate.to_numpy().flatten(),
                        'nstemi_shape':nstemi_shape.to_numpy().flatten(),
                        'nstemi_temporal':nstemi_temporal.to_numpy().flatten(),
                        'nstemi_r2':nstemi_r2.to_numpy().flatten(),})

#Vamos a escalar los datos:
scaler = MinMaxScaler(feature_range=(0,1))
data_scaler_acs = scaler.fit_transform(data_acs)
data_scaler_acs= pd.DataFrame(data_scaler_acs,columns = data_acs.columns)

data_scaler_stemi = scaler.fit_transform(data_stemi)
data_scaler_stemi= pd.DataFrame(data_scaler_stemi,columns = data_stemi.columns)

data_scaler_nstemi = scaler.fit_transform(data_nstemi)
data_scaler_nstemi= pd.DataFrame(data_scaler_nstemi,columns = data_nstemi.columns)

fig = plt.figure(figsize=(5,8)) #x,y
plt.subplots_adjust(wspace=0.4,hspace=0.3)
ax1 = plt.subplot(3, 1, 1)
ax1.boxplot([data_scaler_acs.acs_rmse, data_scaler_acs.acs_r2], #data_scaler_acs.acs_rmse, data_scaler_acs.acs_dilate, data_scaler_acs.acs_shape, data_scaler_acs.acs_temporal,data_scaler_acs.acs_r2
            labels=['RMSE','R2'])
ax1.set_title('Subpoblación ACS',fontsize=10)
ax1.set_ylabel('Rendimiento')
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

ax2 = plt.subplot(3, 1, 2)
ax2.boxplot([data_scaler_stemi.stemi_rmse, data_scaler_stemi.stemi_r2], 
            labels=['RMSE','R2'])
ax2.set_title('Subpoblación STEMI',fontsize=10)
ax2.set_ylabel('Rendimiento')
ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

ax3 = plt.subplot(3, 1, 3)
ax3.boxplot([data_scaler_nstemi.nstemi_rmse, data_scaler_nstemi.nstemi_r2], 
            labels=['RMSE','R2'])
ax3.set_title('Subpoblación NSTEMI',fontsize=10)
ax3.set_ylabel('Rendimiento')
ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

plt.suptitle("Rendimiento del modelo en subpoblaciones de pacientes \n (Redes condicionada - SHAPE)")

filename = "boxplot_test_set_dilate_cond"
plt.tight_layout()
plt.savefig('../experimentos/plots_comparativos/'+str(filename)+'.pdf')
plt.close()