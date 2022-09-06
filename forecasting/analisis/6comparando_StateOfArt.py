"""
Función para calcular la métrica de RMSE en todos los modelos con mejor rendimiento

Para las condicionales fue: Dilate
Para las no condicionales fue: RMSE

ambas con una ventana de predicción de 24 horas
"""

import pandas as pd
import numpy as np
import torch
from torchmetrics import R2Score
from matplotlib import pyplot as plt

#-----------------settings--------------"
patients = "nstemi"
path = "../../experimentos/0estado_arte/"
feature_list = ['arterial_bp_mean','respiratory_rate', 'diastolic_bp','spo2','heart_rate', 'systolic_bp', 'temperature']
num_features = len(feature_list)
seq_length = 6

#----------------------------------------

def r2_corr(pred,true):
    return r2_score(true, pred).mean()

def rmse_loss(pred,true):
    "Computes the root mean square error"
    criterion = torch.nn.MSELoss()
    rmse = torch.sqrt(criterion(pred,true)) 
    return rmse

def get_values_from_mask(mask,values,seqlen, num_features):
    """ Para calcular la pérdida, necesito usar la máscara, entonces
        únicamente se obtienen los valores cuando mask[i]==True
        con ese array que regresa se calculará la pérdida
    """
    array = np.empty(seqlen*num_features)

    for i in range(len(mask)):
        if mask[i]==True:
            array[i] = values[i]
        else:
            array[i]=0 #si es false, es impute_saits, solo para fines prácticos se imputa con 0

    array = array.reshape(seqlen, num_features)
    from_array_to_tensor = torch.tensor(array, dtype=torch.float32)
    return from_array_to_tensor


def computing_rmse_loss_and_r2score():
    print("* Opening ARIMA")
    pred_arima=pd.read_csv('../../experimentos/0estado_arte/arima_'+str(patients)+'/ypred.csv')
    pred_arima=pred_arima[feature_list]
    print("pred_arima:",pred_arima.shape)

    print("* Opening RCGAN")
    pred_rcgan=pd.read_csv('../../experimentos/0estado_arte/RCGAN_cond_'+str(patients)+'/data/ypred.csv')
    pred_rcgan=pred_rcgan[feature_list]
    print("pred_rcgan:",pred_rcgan.shape)

    print("* Opening RGAN")
    pred_rgan=pd.read_csv('../../experimentos/0estado_arte/RGAN_'+str(patients)+'/data/ypred.csv')
    pred_rgan=pred_rgan[feature_list]
    print("pred_rgan:",pred_rgan.shape)

    print("* Opening TIMEGAN")
    pred_timegan=pd.read_csv('../../experimentos/0estado_arte/timegan_'+str(patients)+'/ypred.csv')
    pred_timegan=pred_timegan[feature_list]
    print("pred_timegan:",pred_timegan.shape)

    print("* Opening SeriesNet")
    pred_seriesnet = pd.read_csv('../../experimentos/0estado_arte/seriesnet_'+str(patients)+'/rmse.txt',delimiter=',')

    print("* Opening CardioSeries")
    pred_cardioseries = pd.read_csv('../../experimentos/0estado_arte/cardioseries_'+str(patients)+'/rmse.txt',delimiter=',')

    print("* Opening Mask")
    if patients == "stemi":
        mask=pd.read_csv('../../experimentos/0estado_arte/test_mask_stemi_saits.csv')
        mask=mask[feature_list] #(3120, 7)

        ytrue=np.loadtxt('../../experimentos/0estado_arte/cardioseries_stemi/yreal.txt',delimiter=',')
        ytrue = np.array(ytrue).reshape(-1*6,7)
        ytrue = pd.DataFrame(ytrue, columns = [feature_list]) # (1560, 7)

    else: #nstemi
        mask=pd.read_csv('../../experimentos/0estado_arte/test_mask_nstemi_saits.csv')
        mask=mask[feature_list]

        ytrue=np.loadtxt('../../experimentos/0estado_arte/cardioseries_nstemi/yreal.txt',delimiter=',')
        ytrue = np.array(ytrue).reshape(-1*6,7)
        ytrue = pd.DataFrame(ytrue, columns = [feature_list])

    
    #--- convert from dataframe to numpy ---#
    pred_arima= np.array(pred_arima).reshape(-1,seq_length,num_features)#(1560, 7)
    pred_rcgan = np.array(pred_rcgan).reshape(-1,seq_length,num_features)#(1560, 7)
    pred_rgan = np.array(pred_rgan).reshape(-1,seq_length,num_features)#(1560, 7)
    pred_timegan = np.array(pred_timegan).reshape(-1,seq_length,num_features)#(1560, 7)
    ytrue = np.array(ytrue).reshape(-1,seq_length,num_features)#(3120, 7)
    mask = np.array(mask).reshape(-1,seq_length*2,num_features)#(1560, 7)

    #--- convert from numpy to tensor ---#
    arima_tensor = torch.tensor(pred_arima, dtype=torch.float32) #(260, 6, 7)
    rcgan_tensor = torch.tensor(pred_rcgan, dtype=torch.float32) #(260, 6, 7)
    rgan_tensor = torch.tensor(pred_rgan, dtype=torch.float32) #(260, 6, 7)
    timegan_tensor = torch.tensor(pred_timegan, dtype=torch.float32) #(260, 6, 7)
    ytrue_tensor = torch.tensor(ytrue, dtype=torch.float32) #(260, 6, 7)
    mask_tensor = torch.tensor(mask, dtype=torch.float32) #(260, 6, 7)

    arima_loss,rcgan_loss,rgan_loss,timegan_loss,cardioseries_loss=[],[],[],[],[]
    arima_r2,rcgan_r2,rgan_r2,timegan_r2,cardioseries_r2=[],[],[],[],[]

    r2score = R2Score()#.to(device)

    for index in range(ytrue_tensor.shape[0]): #recorriendo cada paciente
        expected = ytrue_tensor[index,:, :].flatten() #(42)
        arima = arima_tensor[index,:,:].flatten() #(42)
        rcgan = rcgan_tensor[index,:,:].flatten() #(42)
        rgan = rgan_tensor[index,:,:].flatten() #(42)
        timegan = timegan_tensor[index,:,:].flatten() #(42)
        
        """
        mask_tensor[index] #(42) #devuelve la máscara para los 12 pasos (es decir las 48 horas)
        luego se hace una máscara temporal, para seleccionar el y_true (del paso 6 al 12)
        """

        temp_mask =  mask_tensor[index] #
        mask_pat = temp_mask[6:].flatten()

        ytrue_np = get_values_from_mask(mask_pat,expected,seq_length,num_features)
        arima_np = get_values_from_mask(mask_pat,arima,seq_length,num_features)      
        rcgan_array = get_values_from_mask(mask_pat,rcgan,seq_length,num_features)
        rgan_array = get_values_from_mask(mask_pat,rgan,seq_length,num_features)
        timegan_array = get_values_from_mask(mask_pat,timegan,seq_length,num_features)
        ytrue_np = ytrue_np.clone().detach()

        #Computing rmse_loss per patient
        arima_loss.append(rmse_loss(arima_np,ytrue_np).item()/32)
        rcgan_loss.append(rmse_loss(rcgan_array,ytrue_np).item()/32)
        rgan_loss.append(rmse_loss(rgan_array,ytrue_np).item()/32)
        timegan_loss.append(rmse_loss(timegan_array,ytrue_np).item()/32)

        #Computing r2 per patient
        #arima_r2.append(r2score(arima_np.reshape(-1),ytrue_np.reshape(-1)).item())

    print("****** Results (RMSE) ******")
    print("ARIMA:", len(arima_loss),round(np.mean(arima_loss),2),"+-",round(np.std(arima_loss),2))
    print("RCGAN:", len(rcgan_loss),round(np.mean(rcgan_loss),2),"+-",round(np.std(rcgan_loss),2))
    print("RGAN:", len(rgan_loss),round(np.mean(rgan_loss),2),"+-",round(np.std(rgan_loss),2))
    print("TIMEGAN:", len(timegan_loss),round(np.mean(timegan_loss),2),"+-",round(np.std(timegan_loss),2))

    cardioseries_loss= pd.DataFrame({'cardioseries_loss':pred_cardioseries.to_numpy().flatten()})
    pred_seriesnet = pd.DataFrame({'pred_seriesnet':pred_seriesnet.to_numpy().flatten()})

    print("CardioSeries:", round(np.mean(pred_cardioseries.to_numpy().flatten()),2),"+-",round(np.std(pred_cardioseries.to_numpy().flatten()),2))
    print("Seriesnet:", round(np.mean(pred_seriesnet.to_numpy().flatten()),2),"+-",round(np.std(pred_seriesnet.to_numpy().flatten()),2))

    arima_loss = pd.DataFrame({'arima_loss':arima_loss})
    rcgan_loss = pd.DataFrame({'rcgan_loss':rcgan_loss})
    rgan_loss = pd.DataFrame({'rgan_loss':rgan_loss})
    timegan_loss = pd.DataFrame({'timegan_loss':timegan_loss})

    fig = plt.figure(figsize=(8,5)) #x,y
    plt.subplots_adjust(wspace=0.5,hspace=0.3)

    ax1 = plt.subplot(2, 3, 1)
    ax1.boxplot([arima_loss.arima_loss],labels=['Modelo'])
    ax1.set_title('Arima',fontsize=10)
    ax1.set_ylabel('rmse')
    #ax1.set_yticks(list(np.arange(1.5, 3.6, 0.2))) #rango
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax2 = plt.subplot(2, 3, 2)
    ax2.boxplot([rgan_loss.rgan_loss],labels=['Modelo'])
    ax2.set_title('RGAN',fontsize=10)
    ax2.set_ylabel('rmse')
    #ax2.set_yticks(list(np.arange(1.5, 3.6, 0.2))) #rango
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax3 = plt.subplot(2, 3, 3)
    ax3.boxplot([timegan_loss.timegan_loss],labels=['Modelo'])
    ax3.set_title('TIMEGAN',fontsize=10)
    ax3.set_ylabel('rmse')
    #ax3.set_yticks(list(np.arange(1.5, 3.6, 0.2))) #rango
    ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax4 = plt.subplot(2, 3, 4)
    ax4.boxplot([pred_seriesnet.pred_seriesnet],labels=['Modelo'])
    ax4.set_title('Seriesnet',fontsize=10)
    ax4.set_ylabel('rmse')
    #ax4.set_yticks(list(np.arange(0, 1.2, 0.1))) #rango
    ax4.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax5 = plt.subplot(2, 3, 5)
    ax5.boxplot([rcgan_loss.rcgan_loss],labels=['Modelo'])
    ax5.set_title('RCGAN',fontsize=10)
    ax5.set_ylabel('rmse')
    #ax5.set_yticks(list(np.arange(1.5, 3.6, 0.2))) #rango
    ax5.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax6 = plt.subplot(2, 3, 6)
    ax6.boxplot([cardioseries_loss.cardioseries_loss],labels=['Modelo'])
    ax6.set_title('CardioSeries',fontsize=10)
    ax6.set_ylabel('rmse')
    #ax6.set_yticks(list(np.arange(0, 1.2, 0.1))) #rango
    ax6.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    plt.tight_layout()
    plt.savefig('../../experimentos/0plots/estado_arte_'+str(patients)+'.pdf')
    plt.close()            



def plotting_comparative():
    print("---------- Generando box plot--------------------")

    # Después de calcular la rmse_loss y R2 por cada subpoblación (STEMI, NSTEMI,ACS)
    # entonces generamos el box plot

    model_dilate_loss_rmse_acs = pd.read_csv('../experimentos/causal_in6_out6_c5_dilate/metrics_all/metrics_acs_all/model_dilate_loss_rmse.txt', sep=" ")
    model_rcgan_loss_rmse_acs = pd.read_csv('../experimentos/causal_in6_out6_c5_dilate/metrics_all/metrics_acs_all/model_rcgan_loss_rmse.txt', sep=" ")

    model_dilate_loss_rmse_stemi = pd.read_csv('../experimentos/causal_in6_out6_c5_dilate/metrics_all/metrics_stemi_all/model_dilate_loss_rmse.txt', sep=" ")
    model_rcgan_loss_rmse_stemi = pd.read_csv('../experimentos/causal_in6_out6_c5_dilate/metrics_all/metrics_stemi_all/model_rcgan_loss_rmse.txt', sep=" ")

    model_dilate_loss_rmse_nstemi = pd.read_csv('../experimentos/causal_in6_out6_c5_dilate/metrics_all/metrics_nstemi_all/model_dilate_loss_rmse.txt', sep=" ")
    model_rcgan_loss_rmse_nstemi = pd.read_csv('../experimentos/causal_in6_out6_c5_dilate/metrics_all/metrics_nstemi_all/model_rcgan_loss_rmse.txt', sep=" ")

    data_acs = pd.DataFrame({'Dilate_rmseloss':model_dilate_loss_rmse_acs.to_numpy().flatten(),
                         'RCGAN_rmseloss':model_rcgan_loss_rmse_acs.to_numpy().flatten(),})


    data_stemi = pd.DataFrame({'Dilate_rmseloss':model_dilate_loss_rmse_stemi.to_numpy().flatten(),
                         'RCGAN_rmseloss':model_rcgan_loss_rmse_stemi.to_numpy().flatten(),})               

    data_nstemi = pd.DataFrame({'Dilate_rmseloss':model_dilate_loss_rmse_nstemi.to_numpy().flatten(),
                         'RCGAN_rmseloss':model_rcgan_loss_rmse_nstemi.to_numpy().flatten(),})
     

    fig = plt.figure(figsize=(5,8)) #x,y
    plt.subplots_adjust(wspace=0.4,hspace=0.3)
    ax1 = plt.subplot(3, 1, 1)
    ax1.boxplot([data_acs.RCGAN_rmseloss, data_acs.Dilate_rmseloss], 
                labels=["RCGAN","Dilate"])
    ax1.set_title('Subpoblación ACS',fontsize=10)
    ax1.set_ylabel('rmse')
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax2 = plt.subplot(3, 1, 2)
    ax2.boxplot([data_stemi.RCGAN_rmseloss, data_stemi.Dilate_rmseloss], 
                labels=["RCGAN","Dilate"])
    ax2.set_title('Subpoblación STEMI',fontsize=10)
    ax2.set_ylabel('rmse')
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax3 = plt.subplot(3, 1, 3)
    ax3.boxplot([data_nstemi.RCGAN_rmseloss, data_nstemi.Dilate_rmseloss], 
                labels=["RCGAN","Dilate"])
    ax3.set_title('Subpoblación NSTEMI',fontsize=10)
    ax3.set_ylabel('rmse')
    ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)


    #plt.suptitle("Comparativa con el estado del arte \n (Redes condicionadas)")

    filename = "boxplot_estado_arte_condicionadas"

    #plt.savefig('../experimentos/plots_comparativos/'+str(filename)+'.pdf')
    #plt.close()
    plt.show()

"Función para calcular RMSE y R2 por cada modelo"
computing_rmse_loss_and_r2score()

#"Función para graficar el diagrama de caja"
#plotting_comparative()