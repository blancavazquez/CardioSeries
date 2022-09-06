#Predicción de mortalidad por subpoblación
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

#------ Settings --------
filename = "mortalidad_stemi_nstemi"
#-------------------------

stemi_trtr = pd.read_csv('../../experimentos/0cardioseries_rmse/metrics_all/metrics_stemi/trtr.csv')
stemi_tstr = pd.read_csv('../../experimentos/0cardioseries_rmse/metrics_all/metrics_stemi/tstr.csv')
stemi_trts = pd.read_csv('../../experimentos/0cardioseries_rmse/metrics_all/metrics_stemi/trts.csv')

nstemi_trtr = pd.read_csv('../../experimentos/0cardioseries_rmse/metrics_all/metrics_nstemi/trtr.csv')
nstemi_tstr = pd.read_csv('../../experimentos/0cardioseries_rmse/metrics_all/metrics_nstemi/tstr.csv')
nstemi_trts = pd.read_csv('../../experimentos/0cardioseries_rmse/metrics_all/metrics_nstemi/trts.csv')

#---------------------------
stemi_data = pd.DataFrame({
    "ACC":[stemi_trtr.accuracy.values[0],
          stemi_tstr.accuracy.values[0],
          stemi_trts.accuracy.values[0]]}, 
    index=["TrainReal-TestReal","TrainSynthetic-TestReal", "TrainReal-TestSynthetic"])


nstemi_data = pd.DataFrame({
    "ACC":[nstemi_trtr.accuracy.values[0],
          nstemi_tstr.accuracy.values[0],
          nstemi_trts.accuracy.values[0]]}, 
    index=["TrainReal-TestReal","TrainSynthetic-TestReal", "TrainReal-TestSynthetic"])


fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(7,7))
plt.subplots_adjust(wspace=0.2,hspace=0.3)

stemi_data.plot.bar(ax=axes[0],rot=0,title = "Subpoblación STEMI",legend=False,ylabel="Precisión", color= "skyblue").grid(
                True, linestyle='-', which='major', color='lightgrey',alpha=0.2)
nstemi_data.plot.bar(ax=axes[1],rot=0,title = "Subpoblación NSTEMI",legend=False,ylabel="Precisión", color = "lightgrey").grid(
                True, linestyle='-', which='major', color='lightgrey',alpha=0.2)
plt.tight_layout()
plt.savefig('../../experimentos/0plots/'+str(filename)+'.pdf')
plt.close()