"""
Code for plotting the comparative with all imputation methods (TIME)
Author: Blanca Vázquez
"""

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import datetime 

params = {'axes.labelsize': 10,'xtick.labelsize':10,'ytick.labelsize':10,
          'axes.titlesize': 10,'legend.fontsize':10}
plt.rcParams.update(params)

resultados = pd.read_csv("data_kfold/resultados.csv")
filename = "comparativo_time"
resultados["minutos"] = resultados['tiempo'] / 60 #convertir de segundos a minutos


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
plt.subplots_adjust(wspace=0.4,hspace=0.6)

sns.lineplot(x='porcentaje', y='minutos', data=simple, ax = axes[0,0],color='green')
axes[0,0].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[0,0].set_title('Simple')
axes[0,0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='minutos', data=iterative, ax = axes[0,1],color='green')
axes[0,1].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[0,1].set_title('Iterative')
axes[0,1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='minutos', data=interpolate_linear, ax = axes[0,2],color='green')
axes[0,2].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[0,2].set_title('Linear interpolation')
axes[0,2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='minutos', data=knn, ax = axes[1,0],color='green')
axes[1,0].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[1,0].set_title('k-Nearest Neighbors')
axes[1,0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='minutos', data=random, ax = axes[1,1],color='green')
axes[1,1].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[1,1].set_title('Random')
axes[1,1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='minutos', data=mice, ax = axes[1,2],color='green')
axes[1,2].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[1,2].set_title('MICE')
axes[1,2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='minutos', data=expectation_maximization, ax = axes[2,0],color='green')
axes[2,0].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[2,0].set_title('Expectation maximization')
axes[2,0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='minutos', data=moving_window, ax = axes[2,1],color='green')
axes[2,1].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[2,1].set_title('Moving average window')
axes[2,1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

sns.lineplot(x='porcentaje', y='minutos', data=saits, ax = axes[2,2],color='green')
axes[2,2].set_xticks([10, 20, 30, 40, 50])
axes[0,0].set_xlabel('Missing rate (%)')
axes[2,2].set_ylabel('Time (minutes)')
axes[2,2].set_title('SAITS')
axes[2,2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

plt.savefig('graficas/'+str(filename)+'.png', dpi=300, bbox_inches='tight')
plt.savefig('graficas/'+str(filename)+'.pdf', dpi=300, bbox_inches='tight')
plt.close()