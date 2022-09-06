import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

#------ Settings --------
title = "Rendimientos de los modelos en el conjunto de validación \n por función de activación (Módulo condicional - BiLSTM)"
filename = "boxplot_dilate_in6_out6_c5_bilstm_optimizadores_RMSE"
path = "optimizadores_rmse/sgd_"
#-------------------------

elu = pd.read_csv('../../experimentos/'+str(path)+'elu'+'/metrics/rmse_val.txt', sep=" ",names=["elu"])
gelu = pd.read_csv('../../experimentos/'+str(path)+'gelu'+'/metrics/rmse_val.txt', sep=" ",names=["gelu"])
selu = pd.read_csv('../../experimentos/'+str(path)+'selu'+'/metrics/rmse_val.txt', sep=" ",names=["selu"])
leaky_relu= pd.read_csv('../../experimentos/'+str(path)+'leaky'+'/metrics/rmse_val.txt', sep=" ",names=["leaky_relu"])
mish = pd.read_csv('../../experimentos/'+str(path)+'mish'+'/metrics/rmse_val.txt', sep=" ",names=["mish"])
print("elu:",elu.shape,"gelu:",gelu.shape,"selu:",selu.shape,"leaky_relu:",leaky_relu.shape,"mish:",mish.shape)

fig, ax = plt.subplots(figsize=(8,5))
ax.boxplot([elu.elu,gelu.gelu,selu.selu,leaky_relu.leaky_relu,mish.mish], labels=['ELU','GELU','SELU','LeakyRelu','Mish'])
ax.set(
    axisbelow=True,  # Hide the grid behind plot objects
    title=title,
    xlabel='Funciones de pérdida evaluadas',
    ylabel='RMSE')
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

#rango =list(np.arange(0, 2.0, 0.2))
#ax.set_yticks(rango)

#plt.tight_layout()
#plt.savefig('../../experimentos/0plots/'+'boxplot'+str(path)+'.pdf')
#plt.close()

plt.show()

#----------------------------#
#----------------------------#


# fig = plt.figure(figsize=(8,9)) #x,y
# plt.subplots_adjust(wspace=0.4,hspace=0.3)

# ax1 = plt.subplot(4, 1, 1)
# ax1.boxplot([elu_rms.elu,gelu_rms.gelu,selu_rms.selu,leaky_rms.leaky_relu,mish_rms.mish], labels=['ELU','GELU','SELU','LeakyRelu','Mish'])
# ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
# ax1.set_yticks(list(np.arange(0, 2, 0.20)) )
# ax1.set_ylabel('RMSE')
# ax1.set_title("Optimizador RMSProp")

# ax2 = plt.subplot(4, 1, 2)
# ax2.boxplot([elu_sgd.elu,gelu_sgd.gelu,selu_sgd.selu,leaky_sgd.leaky_relu,mish_sgd.mish], labels=['ELU','GELU','SELU','LeakyRelu','Mish'])
# ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
# ax2.set_yticks(list(np.arange(0, 0.8, 0.20)) )
# ax2.set_ylabel('RMSE')
# ax2.set_title("Optimizador SGD")

# ax3 = plt.subplot(4, 1, 3) #elu_adam.elu,
# ax3.boxplot([gelu_adam.gelu,selu_adam.selu,leaky_adam.leaky_relu,mish_adam.mish], labels=['GELU','SELU','LeakyRelu','Mish'])
# ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
# ax3.set_ylabel('RMSE')
# ax3.set_title("Optimizador Adam")

# ax4 = plt.subplot(4, 1, 4)
# ax4.boxplot([elu_adamw.elu,gelu_adamw.gelu,selu_adamw.selu,leaky_adamw.leaky_relu,mish_adamw.mish], labels=['ELU','GELU','SELU','LeakyRelu','Mish'])
# ax4.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
# ax4.set_yticks(list(np.arange(0, 0.8, 0.20)) )
# ax4.set_ylabel('RMSE')
# ax4.set_title("Optimizador AdamW")

# plt.suptitle(title)
# plt.tight_layout()
# #plt.show()
# plt.savefig('../../experimentos/0plots/'+str(filename)+'.png')
# plt.savefig('../../experimentos/0plots/'+str(filename)+'.pdf')
# plt.close()
