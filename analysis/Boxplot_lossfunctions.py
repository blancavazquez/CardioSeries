"""
Code for plotting a boxplot with all trained models (loss functions)
Author: Blanca Vázquez
"""

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

#------ Settings --------
title = "Tamaño de la ventana de predicción = 12 horas"
path = "_in9_out3_c0_5x10"
#-------------------------

#Opening loss_on_valset
model_rmse = pd.read_csv('../../experimentos/'+'rmse'+str(path)+'/metrics/rmse_val.txt', sep=" ")
model_mae = pd.read_csv('../../experimentos/'+'mae'+str(path)+'/metrics/rmse_val.txt', sep=" ")
model_mmd = pd.read_csv('../../experimentos/'+'mmd'+str(path)+'/metrics/rmse_val.txt', sep=" ")
model_dilate = pd.read_csv('../../experimentos/'+'dilate'+str(path)+'/metrics/rmse_val.txt', sep=" ")
model_shape = pd.read_csv('../../experimentos/'+'shape'+str(path)+'/metrics/rmse_val.txt', sep=" ")
model_temporal = pd.read_csv('../../experimentos/'+'temporal'+str(path)+'/metrics/rmse_val.txt', sep=" ")

data = pd.DataFrame({'RMSE':model_rmse.to_numpy().flatten(),
					 'MAE':model_mae.to_numpy().flatten(),
					 'MMD':model_mmd.to_numpy().flatten(),
					 'Dilate':model_dilate.to_numpy().flatten(),
					 'Shape':model_shape.to_numpy().flatten(),
					 'Temporal':model_temporal.to_numpy().flatten()
					 })

fig, ax = plt.subplots(figsize=(8,5))
ax.boxplot(data, labels = ["RMSE","MAE","MMD","DILATE","SHAPE","TEMPORAL"],showfliers=True)
ax.set(
    axisbelow=True,  # Hide the grid behind plot objects
    title=title,
    xlabel='Loss functions (evaluated)',
    ylabel='RMSE')
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

rango =list(np.arange(0, 1.0, 0.1))
ax.set_yticks(rango)

plt.tight_layout()
plt.savefig('../../experimentos/0plots/'+'boxplot'+str(path)+'.pdf')
plt.close()