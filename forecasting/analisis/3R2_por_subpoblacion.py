#Predicción de mortalidad por subpoblación
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib import cm

#------ Settings --------
filename = "comparativa_por subpoblacion_rmse_r2_cond_dilate"
path = "causal_in6_out6_c5_dilate"
function = "DILATE"
#-------------------------

#---- Cargando RMSE por cada subpoblación
acs_women_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_women/rmse.txt', sep=" ")
acs_men_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_men/rmse.txt', sep=" ")
acs_hipertension_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_hipertension/rmse.txt', sep=" ")
acs_diabetes_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_diabetes/rmse.txt', sep=" ")
acs_mayor60_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_mayor60/rmse.txt', sep=" ")
acs_menor60_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_menor60/rmse.txt', sep=" ")

stemi_women_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_women/rmse.txt', sep=" ")
stemi_men_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_men/rmse.txt', sep=" ")
stemi_hipertension_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_hipertension/rmse.txt', sep=" ")
stemi_diabetes_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_diabetes/rmse.txt', sep=" ")
stemi_mayor60_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_mayor60/rmse.txt', sep=" ")
stemi_menor60_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_menor60/rmse.txt', sep=" ")

nstemi_women_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_women/rmse.txt', sep=" ")
nstemi_men_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_men/rmse.txt', sep=" ")
nstemi_hipertension_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_hipertension/rmse.txt', sep=" ")
nstemi_diabetes_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_diabetes/rmse.txt', sep=" ")
nstemi_mayor60_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_mayor60/rmse.txt', sep=" ")
nstemi_menor60_rmse = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_menor60/rmse.txt', sep=" ")

#---- Cargando R2 por cada subpoblación
acs_women_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_women/r2.txt', sep=" ")
acs_men_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_men/r2.txt', sep=" ")
acs_hipertension_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_hipertension/r2.txt', sep=" ")
acs_diabetes_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_diabetes/r2.txt', sep=" ")
acs_mayor60_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_mayor60/r2.txt', sep=" ")
acs_menor60_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_acs_menor60/r2.txt', sep=" ")

stemi_women_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_women/r2.txt', sep=" ")
stemi_men_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_men/r2.txt', sep=" ")
stemi_hipertension_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_hipertension/r2.txt', sep=" ")
stemi_diabetes_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_diabetes/r2.txt', sep=" ")
stemi_mayor60_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_mayor60/r2.txt', sep=" ")
stemi_menor60_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_stemi_menor60/r2.txt', sep=" ")

nstemi_women_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_women/r2.txt', sep=" ")
nstemi_men_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_men/r2.txt', sep=" ")
nstemi_hipertension_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_hipertension/r2.txt', sep=" ")
nstemi_diabetes_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_diabetes/r2.txt', sep=" ")
nstemi_mayor60_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_mayor60/r2.txt', sep=" ")
nstemi_menor60_r2 = pd.read_csv('../experimentos/'+str(path)+'/metrics_all/metrics_nstemi_menor60/r2.txt', sep=" ")

data_rmse = pd.DataFrame({
    "ACS":[acs_women_rmse.mean().item(),
           acs_men_rmse.mean().item(), 
           acs_hipertension_rmse.mean().item(), 
           acs_diabetes_rmse.mean().item(), 
           acs_mayor60_rmse.mean().item(),
           acs_menor60_rmse.mean().item()],
    "STEMI":[stemi_women_rmse.mean().item(), 
             stemi_men_rmse.mean().item(), 
             stemi_hipertension_rmse.mean().item(), 
             stemi_diabetes_rmse.mean().item(),
             stemi_mayor60_rmse.mean().item(),
             stemi_menor60_rmse.mean().item()],
    "NSTEMI":[nstemi_women_rmse.mean().item(), 
             nstemi_men_rmse.mean().item(), 
             nstemi_hipertension_rmse.mean().item(), 
             nstemi_diabetes_rmse.mean().item(), 
             nstemi_mayor60_rmse.mean().item(),
             nstemi_menor60_rmse.mean().item()]}, 
    index=["Mujeres", "Hombres", "Hipertension", "Diabetes", "Mayor 60 años","Menor 60 años"])

data_r2 = pd.DataFrame({
    "ACS":[acs_women_r2.mean().item(),
           acs_men_r2.mean().item(), 
           acs_hipertension_r2.mean().item(), 
           acs_diabetes_r2.mean().item(), 
           acs_mayor60_r2.mean().item(),
           acs_menor60_r2.mean().item()],
    "STEMI":[stemi_women_r2.mean().item(), 
             stemi_men_r2.mean().item(), 
             stemi_hipertension_r2.mean().item(), 
             stemi_diabetes_r2.mean().item(),
             stemi_mayor60_r2.mean().item(),
             stemi_menor60_r2.mean().item()],
    "NSTEMI":[nstemi_women_r2.mean().item(), 
             nstemi_men_r2.mean().item(), 
             nstemi_hipertension_r2.mean().item(), 
             nstemi_diabetes_r2.mean().item(), 
             nstemi_mayor60_r2.mean().item(),
             nstemi_menor60_r2.mean().item()]}, 
    index=["Mujeres", "Hombres", "Hipertension", "Diabetes", "Mayor 60 años","Menor 60 años"])

#Vamos a escalar los datos:
scaler = MinMaxScaler(feature_range=(0,1))
data_scaler_rmse = scaler.fit_transform(data_rmse)
data_scaler_rmse= pd.DataFrame(data_scaler_rmse,columns = data_rmse.columns,index=data_rmse.index)

data_scaler_r2 = scaler.fit_transform(data_r2)
data_scaler_r2= pd.DataFrame(data_scaler_r2,columns = data_r2.columns,index=data_r2.index)


fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(10,8))
plt.subplots_adjust(wspace=0.2,hspace=0.4)

data_scaler_rmse.plot.bar(ax=axes[0],rot=0,legend=False,ylabel="RMSE").grid(
                True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

data_scaler_r2.plot.bar(ax=axes[1],rot=0,title = "Comparativa por R2",legend=False,ylabel="R2").grid(
                True, linestyle='-', which='major', color='lightgrey',alpha=0.2)

axes[0].set_title('Comparativa en subpoblaciones de pacientes (Red condicionada '+str(function)+')'+'\n'+'Comparativa por RMSE')
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='lower center', shadow=True, ncol=5,borderaxespad=0) #(0.06, 0.2)


plt.tight_layout()
plt.savefig('../experimentos/plots_comparativos/'+str(filename)+'.pdf')
plt.close()