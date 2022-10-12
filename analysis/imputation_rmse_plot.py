"""
Code for plotting the comparative with all imputation methods (RMSE)
Author: Blanca Vázquez
"""

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

params = {'axes.labelsize': 10,'xtick.labelsize':10,'ytick.labelsize':10,
          'axes.titlesize': 10,'legend.fontsize':10}
plt.rcParams.update(params)

resultados = pd.read_csv("data_kfold/resultados.csv")
filename = "comparativo_RMSE"

print("* Plotting")
simple = resultados[(resultados.metodo=="simple")] 
iterative = resultados[(resultados.metodo=="iterative")] 
interpolate_linear = resultados[(resultados.metodo=="interpolate_linear")] 
knn = resultados[(resultados.metodo=="knn")] 
random = resultados[(resultados.metodo=="random")] 
mice = resultados[(resultados.metodo=="mice")] 
expectation_maximization = resultados[(resultados.metodo=="expectation_maximization")] 
moving_window = resultados[(resultados.metodo=="moving_window")] 
saits = resultados[(resultados.metodo=="saits")] 

fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(12,8))
plt.subplots_adjust(wspace=0.3,hspace=0.6)

sns.lineplot(x='porcentaje', y='rmse', data=simple, ax = axes[0,0])
axes[0,0].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[0,0].set_ylabel('RMSE')
axes[0,0].set_title('Simple')
axes[0,0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='rmse', data=iterative, ax = axes[0,1])
axes[0,1].set_xticks([10, 20, 30, 40, 50])
axes[0,1].set_xlabel('Missing rate (%)')
axes[0,1].set_ylabel('RMSE')
axes[0,1].set_title('Iterative')
axes[0,1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='rmse', data=interpolate_linear, ax = axes[0,2])
axes[0,2].set_xticks([10, 20, 30, 40, 50])
axes[0,2].set_xlabel('Missing rate (%)')
axes[0,2].set_ylabel('RMSE')
axes[0,2].set_title('Linear interpolation')
axes[0,2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='rmse', data=knn, ax = axes[1,0])
axes[1,0].set_xticks([10, 20, 30, 40, 50])
axes[1,0].set_xlabel('Missing rate (%)')
axes[1,0].set_ylabel('RMSE')
axes[1,0].set_title('k-Nearest Neighbors')
axes[1,0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='rmse', data=random, ax = axes[1,1])
axes[1,1].set_xticks([10, 20, 30, 40, 50])
axes[1,1].set_xlabel('Missing rate (%)')
axes[1,1].set_ylabel('RMSE')
axes[1,1].set_title('Random')
axes[1,1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='rmse', data=mice, ax = axes[1,2])
axes[1,2].set_xticks([10, 20, 30, 40, 50])
axes[1,2].set_xlabel('Missing rate (%)')
axes[1,2].set_ylabel('RMSE')
axes[1,2].set_title('MICE')
axes[1,2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='rmse', data=expectation_maximization, ax = axes[2,0])
axes[2,0].set_xticks([10, 20, 30, 40, 50])
axes[2,0].set_xlabel('Missing rate (%)')
axes[2,0].set_ylabel('RMSE')
axes[2,0].set_title('Expectation maximization')
axes[2,0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='rmse', data=moving_window, ax = axes[2,1])
axes[2,1].set_xticks([10, 20, 30, 40, 50])
axes[2,1].set_xlabel('Missing rate (%)')
axes[2,1].set_ylabel('RMSE')
axes[2,1].set_title('Moving average window')
axes[2,1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='rmse', data=saits, ax = axes[2,2])
axes[2,2].set_xticks([10, 20, 30, 40, 50])
axes[2,2].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('RMSE')
axes[2,2].set_title('SAITS')
axes[2,2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

#plt.suptitle("Rendimiento de los métodos de imputación")
plt.savefig('plots/'+str(filename)+'.pdf', dpi=300, bbox_inches='tight')
plt.close()