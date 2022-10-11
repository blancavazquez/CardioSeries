"""
Author Blanca Vázquez
"""
import os
import sys
import json
import math
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from time import time
import xgboost as xgb
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score, \
                            roc_auc_score, auc,roc_curve,average_precision_score, \
                            accuracy_score,f1_score, classification_report, \
                            precision_recall_curve

#Loading pytorch_lightning
import torch
from torch.utils.data import DataLoader

#Loading losses
from loss.losses import rmse_loss,r2_corr
warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
loss_func_train_list, loss_func_val_list, rmse_validation = [],[],[]
rmse_val_list,dilate_val_list,shape_val_list,temporal_val_list,r2_val_list = [],[],[],[],[]

def create_folders(foldername, input_seq_length, output_seq_length,conditions):
    """Function for creating folders for saving results"""
    folder= str(foldername)+'_in'+str(input_seq_length)+'_out'+str(output_seq_length)+'_c'+str(conditions)
    if not os.path.isdir('experimentos/'+folder+'/'):
        os.mkdir('experimentos/'+folder+'/')
    if not os.path.isdir('experimentos/'+folder+'/data/'):
        os.mkdir('experimentos/'+folder+'/data/')
    if not os.path.isdir('experimentos/'+folder+'/metrics/'):
        os.mkdir('experimentos/'+folder+'/metrics/')
    if not os.path.isdir('experimentos/'+folder+'/models/'):
        os.mkdir('experimentos/'+folder+'/models/')
    if not os.path.isdir('experimentos/'+folder+'/plots/'):
        os.mkdir('experimentos/'+folder+'/plots/')
    path='experimentos/'+folder+'/'
    return path

def loading_data(feature_list,path):
    """Loading data, splitting in past and targest, and scaling data"""
    train = pd.read_csv("impute/data_ACS/train_sin_acs_saits.csv")
    val = pd.read_csv("impute/data_ACS/val_sin_acs_saits.csv")
    train_mask = pd.read_csv("impute/data_ACS/train_mask_sin_acs_saits.csv")
    val_mask = pd.read_csv("impute/data_ACS/val_mask_sin_acs_saits.csv")

    train = train.iloc[: , 1:]
    val = val.iloc[: , 1:]
    train_mask = train_mask.iloc[: , 1:]
    val_mask = val_mask.iloc[: , 1:]
    train,val = scale_data(train,val,feature_list,path)

    return train,val,train_mask, val_mask

def scale_data(train,val,feature_list,path):
    """Scale data using MinMaxScaler"""
    seq_length = 12

    #extracting pids#
    train_pids, val_pids = train[['pid']],val[['pid']]
    train_data, val_data = train[feature_list], val[feature_list]

    #Reshape data
    Xtrain = np.asarray(train_data).reshape(-1*seq_length,train_data.shape[1])
    Xval = np.asarray(val_data).reshape(-1*seq_length,val_data.shape[1])

    #Defining "MinMaxScaler"
    scale_range=(-1, 1) #-1,1
    scaler = MinMaxScaler(feature_range=scale_range).fit(np.vstack([Xtrain])) #training
    pickle.dump(scaler, open(path+'data/'+'scaler', 'wb')) #Saving scaler

    scaled_train = scaler.transform(Xtrain).reshape(-1*train_data.shape[0],train_data.shape[1])
    scaled_val = scaler.transform(Xval).reshape(-1* val_data.shape[0],val_data.shape[1])
    pd_train = pd.DataFrame(scaled_train,columns = train_data.columns)
    pd_val = pd.DataFrame(scaled_val,columns = val_data.columns)

    #concat scaled data + pids
    pd_train = pd.concat([train_pids, pd_train], axis=1)
    pd_val = pd.concat([val_pids, pd_val], axis=1)
    return pd_train,pd_val

def extraction_demo_data():
    """ Function for extracting demographic data (training)"""
    demo = pd.read_csv("impute/data/mimic_demo_diagnoses.csv")

    #age & sex
    demo['age'] = round(demo.AGE,1)
    demo['gender'] = np.where(demo['GENDER'] == 'M', 1, 0).astype(int)

    #comorbilities
    #join: Cardiac dysrhythmias + Cardiac arrest and ventricular fibrillation
    demo['heart_diseases'] = demo['Cardiac dysrhythmias'] + demo['Cardiac arrest and ventricular fibrillation'] 
    demo['hypertension'] = round(demo['Hypertension'],1)
    demo['diabetes'] = round(demo['Diabetes mellitus'],1)
    demo['heart_diseases'] = np.where(demo['heart_diseases']>= 1, 1, 0).astype(int)

    #ICU type
    demo['micu'] = np.where(demo['ICU'] == 'MICU', 1, 0).astype(int)
    demo['csru'] = np.where(demo['ICU'] == 'CSRU', 1, 0).astype(int)
    demo['sicu'] = np.where(demo['ICU'] == 'SICU', 1, 0).astype(int)
    demo['ccu'] = np.where(demo['ICU'] == 'CCU', 1, 0).astype(int)
    demo['tsicu'] = np.where(demo['ICU'] == 'TSICU', 1, 0).astype(int)

    train_demo = demo[['PID','age','gender','diabetes','hypertension','heart_diseases',
                       'micu','csru','sicu','ccu','tsicu']]
    val_demo = demo[['PID','age','gender','diabetes','hypertension','heart_diseases',
                    'micu','csru','sicu','ccu','tsicu']]
    return train_demo, val_demo

def rescale_data(np_data,seq_length,num_features,path):
    """ Function for rescaling data"""
    scaler=pickle.load(open(path+'data/'+'scaler','rb'))
    np_array_rescale= scaler.inverse_transform(np_data)
    np_array_rescale = np.around(np_array_rescale,2)
    np_array_rescale = np_array_rescale.reshape(-1,seq_length,num_features)
    return np_array_rescale

def extracting_demo_mimic(test,variables):
    """Function for extracting data from MIMIC (testing)"""
    seq_length = 6
    demo_data = pd.read_csv("impute/data/mimic_demo_diagnoses.csv")
    patients = set(test.pid)

    demo_data=pd.DataFrame(demo_data,columns=['PID','GENDER','AGE','Diabetes mellitus','Hypertension',
                                             'Cardiac dysrhythmias','Cardiac arrest and ventricular fibrillation','ICU'])
    demo = pd.DataFrame(columns=['PID','gender','age','Diabetes mellitus','Hypertension',
                                 'Cardiac dysrhythmias','Cardiac arrest and ventricular fibrillation','ICU'])

    list_remove_patients = []
    for pat in patients:
        data = demo_data[demo_data.PID == pat]
        if data.shape[0] == 1:
            values = {"PID":data.PID.item(),"age":data.AGE.item(),"gender":data.GENDER.item(),
                      "Diabetes":data['Diabetes mellitus'].item(),
                      "Hypertension":data['Hypertension'].item(),
                      "Cardiac":data['Cardiac dysrhythmias'].item(),
                      "Fibrillation":data['Cardiac arrest and ventricular fibrillation'].item(),
                      "ICU":data['ICU'].item()}

            demo = demo.append(values,ignore_index=True) #saving demo data
        else:
            list_remove_patients.append(pat)

    demo['heart_diseases'] = demo['Cardiac'] + demo['Fibrillation']
    demo['heart_diseases'] = np.where(demo['heart_diseases']>= 1, 1, 0).astype(int)
    demo['hypertension'] = round(demo['Hypertension'],1).astype(int)
    demo['diabetes'] = round(demo['Diabetes'],1).astype(int)
    demo['age'] = round(demo.age,1)
    demo['gender'] = np.where(demo['gender'] == 'M', 1, 0).astype(int)
    demo['tsicu'] = np.where(demo['ICU']=='TSICU',1,0).astype(int)
    demo['sicu'] = np.where(demo['ICU']=='SICU',1,0).astype(int)
    demo['micu'] = np.where(demo['ICU']=='MICU',1,0).astype(int)
    demo['ccu'] = np.where(demo['ICU']=='CCU',1,0).astype(int)
    demo['csru'] = np.where(demo['ICU']=='CSRU',1,0).astype(int)

    demo = demo[['PID','age','gender','diabetes','hypertension','heart_diseases',
                 'tsicu','sicu','micu','ccu','csru']]
    for pid in list_remove_patients:
        test = test.drop(test[test.pid == pid].index)    
    print("Patients included:", demo.shape[0], " Patients removed:", len(list_remove_patients))
    pat_demo = set(demo.PID)
    np_test = test.to_numpy().reshape(-1,(seq_length*2)*test.shape[1])
    df_test = pd.DataFrame(np_test)
    df_test.rename(columns={ df_test.columns[0]: "pid" }, inplace = True)
    save_test = pd.DataFrame() 

    for pat in pat_demo:
        data = df_test[df_test.pid == pat]
        save_test = save_test.append(data,ignore_index=True)
    np_save_test = save_test.to_numpy().reshape(-1,test.shape[1])
    df_save_test = pd.DataFrame(np_save_test,columns=test.columns)
    return demo, df_save_test

def saving_logs_training(loss):
    """Function for saving loss (training)"""
    loss_func_train_list.extend(loss)

def saving_logs_validation(loss_val,rmse_val,dilate_val,shape_val,temporal_val,r2_val):
    """Function for saving loss (validation)"""
    loss_func_val_list.extend(loss_val)
    rmse_val_list.extend(rmse_val)
    dilate_val_list.extend(dilate_val)
    shape_val_list.extend(shape_val)
    temporal_val_list.extend(temporal_val)
    r2_val_list.extend(r2_val)

def plotting_losses(path,net):
    """Function plotting losses after training"""
    np.savetxt(path +"metrics/"+'loss_train.txt', loss_func_train_list, delimiter=',')
    np.savetxt(path +"metrics/"+'loss_val.txt', loss_func_val_list, delimiter=',') 
    np.savetxt(path +"metrics/"+'rmse_val.txt', rmse_val_list, delimiter=',') 
    np.savetxt(path +"metrics/"+'dilate_val.txt', dilate_val_list, delimiter=',') 
    np.savetxt(path +"metrics/"+'shape_val.txt', shape_val_list, delimiter=',') 
    np.savetxt(path +"metrics/"+'temporal_val.txt', temporal_val_list, delimiter=',')
    np.savetxt(path +"metrics/"+'r2_val.txt', r2_val_list, delimiter=',')

    dict ={
        "RMSE":str(round(np.mean(rmse_val_list),3))+"+-"+str(round(np.std(rmse_val_list),3)),
        "Dilate":str(round(np.mean(dilate_val_list),3))+"+-"+str(round(np.std(dilate_val_list),3)),
        "Shape":str(round(np.mean(shape_val_list),3))+"+-"+str(round(np.std(shape_val_list),3)),
        "Temporal":str(round(np.mean(temporal_val_list),3))+"+-"+str(round(np.std(temporal_val_list),3)),
        "R2":str(round(np.mean(r2_val_list),3))+"+-"+str(round(np.std(r2_val_list),3))
    }

    pathfile = path +"metrics/"+"performance_on_valset.txt"
    with open(pathfile, 'w') as f:
        print(dict, file=f)
    
    #Plotting train vs val
    fig, ax = plt.subplots(2,figsize=(12, 4))
    ax[0].plot(loss_func_train_list, label="Entrenamiento")
    ax[0].set_title('Pérdida del modelo')
    ax[0].set_ylabel('Pérdida')
    ax[0].legend(loc='upper right')

    ax[1].plot(loss_func_val_list, label = "Validación", color = "red")
    ax[1].set_ylabel('Pérdida')
    ax[1].set_xlabel('Iteraciones')
    ax[1].legend(loc='upper right')
    plt.savefig(path+"metrics/plot_train_vs_val.png")
    plt.close()

    #Showing metrics on valset
    print("* Performance")
    print("Loss (using on training):", round(np.mean(loss_func_val_list),3),"+-",round(np.std(loss_func_val_list),3))
    print("RMSE:", round(np.mean(rmse_val_list),3),"+-",round(np.std(rmse_val_list),3))
    print("Dilate:", round(np.mean(dilate_val_list),3),"+-",round(np.std(dilate_val_list),3))
    print("Shape:", round(np.mean(shape_val_list),3),"+-",round(np.std(shape_val_list),3))
    print("Temporal:", round(np.mean(temporal_val_list),3),"+-",round(np.std(temporal_val_list),3))
    print("R2:", round(np.mean(r2_val_list),3),"+-",round(np.std(r2_val_list),3))

def average_data(data):
    """Function for grouping data"""
    #selección de variables de interés
    feature_list = ["pid","arterial_bp_mean","respiratory_rate","diastolic_bp","spo2",
                    "heart_rate","systolic_bp", "temperature"]

    data = data[feature_list]
    pids = set(data.pid) #unique pids
    new_seq_length = 12 #average each 4 hours (48 timesteps/4avg = 12 timesteps)
    data_array = np.empty(shape=(len(pids),new_seq_length,len(feature_list)))

    i=0
    for pat in pids:
        patient_data = data[data.pid == pat]
        avg = patient_data.values.reshape(-1,4,patient_data.shape[1])
        avg = pd.DataFrame(np.nanmean(avg,axis=1))
        data_array[i,:] = avg
        i+=1

    data_array = np.around(np.reshape(data_array, (len(data_array)*new_seq_length, len(feature_list))),2)
    new_pd = pd.DataFrame(data_array,columns=feature_list) #[470856, 9]
    return new_pd
    

def loading_eicu(seq_length,feature_list,path,missing_values,saits_impute,subpopulation):
    """Function for loading eICU data"""
    if subpopulation == "stemi":
        test_data =pd.read_csv("impute/data_eICU/stemi/samples.csv")
        test_pids =pd.read_csv("impute/data_eICU/stemi/pids.csv")

    if subpopulation == "general":
        test_data =pd.read_csv("impute/data/eicu_data.csv")
        test_pids =pd.read_csv("impute/data/eicu_pids.csv")

    if subpopulation == "nstemi":
        test_data =pd.read_csv("impute/data_eICU/nstemi/samples.csv")
        test_pids =pd.read_csv("impute/data_eICU/nstemi/pids.csv")

    test_pids.rename(columns = {'0':'pid'}, inplace = True)
    lista_pids = test_pids['pid'].values.tolist()
    new_list=[]
    new_list.extend(np.repeat(lista_pids,48))
    test_data.insert(0, 'pid',new_list)
    test = average_data(test_data) #average each 4 hours (48 timesteps/4avg = 12 timesteps)

    mask = torch.ones(test.shape, dtype=torch.int32)
    feature_list = test.columns
    mask = torch.tensor(mask).eq(1).view(-1).detach().cpu().numpy().reshape(-1,test.shape[1])
    mask = pd.DataFrame(mask,columns = feature_list)
    mask.drop('pid', axis=1, inplace=True)
    mask.insert(0, 'pid',test['pid']) ##added pid column

    #extracting pids
    test_pids = test[['pid']]#,'offset']]
    selection_features = feature_list[1:]
    test_data = test[selection_features]

    scale_range=(0, 1)
    scaler = MinMaxScaler(feature_range=scale_range).fit(np.vstack([test_data]))
    pickle.dump(scaler, open(path+'data/'+'scaler_eicu', 'wb'))
    scaled_test = scaler.transform(test_data).reshape(-1*test_data.shape[0],test_data.shape[1])
    pd_test = pd.DataFrame(scaled_test,columns = test_data.columns)
    pd_test = pd.concat([test_pids, pd_test], axis=1)

    return pd_test, mask,selection_features


def loading_data_for_testing(seq_length,feature_names,path,missing_values,saits_impute,subpopulation):
    """Function for loading STEMI & NSTEMI data (testing)"""
    path_data = "../"
    
    if subpopulation == "stemi":
        test=pd.read_csv(path_data+"test_stemi_saits.csv")
        test_mask=pd.read_csv(path_data+"test_mask_stemi_saits.csv")
    elif subpopulation == "nstemi":
        test=pd.read_csv(path_data+"test_nstemi_saits.csv")
        test_mask=pd.read_csv(path_data+"test_mask_nstemi_saits.csv") 
    else: 
        test=pd.read_csv(path_data+"val_sin_acs_saits.csv")
        test_mask=pd.read_csv(path_data+"val_mask_sin_acs_saits.csv")

    feature_names = feature_names
    feature_names.insert(0,'pid') #adding pid to feature_names (first position)
    test = test[feature_names]
    test_mask = test_mask[feature_names]
    
    #extracting pids
    test_pids = test[['pid']]#,'offset']]
    selection_features = feature_names[1:]
    test_data = test[selection_features]

    #Reshaping data
    Xtest = np.asarray(test_data).reshape(-1*seq_length,test_data.shape[1])
    scaler=pickle.load(open(path+'data/'+'scaler','rb'))
    scaled_test = scaler.transform(Xtest).reshape(-1* test_data.shape[0],test_data.shape[1])
    pd_test = pd.DataFrame(scaled_test,columns = test_data.columns)

    #concat scaled data + pids
    pd_test = pd.concat([test_pids, pd_test], axis=1)
    return pd_test, test_mask,selection_features

def rmse(ytrue,ypred):
    """ Function for computing rmse """
    mse = mean_squared_error(ytrue,ypred)
    return math.sqrt(mse)

def saving_files_for_mortality(data_real,data_pred,features,path):
    """ Function for saving y_true and y_pred (used for training a mortality prediction model)"""
    data_real = np.array(data_real).reshape(-1,1)
    np.savetxt(path +"metrics/"+'yreal.txt', data_real, delimiter=',')
    data_pred = np.array(data_pred).reshape(-1,1)
    np.savetxt(path +"metrics/"+'ypred.txt', data_pred, delimiter=',')

def saving_metrics(list_rmse,list_dilate,list_shape,list_temporal,list_r2,path):
    """ Function for saving metrics (testing) """
    print("RMSE:", round(np.mean(list_rmse),3),"+-",round(np.std(list_rmse),3))
    print("Dilate:", round(np.mean(list_dilate),3),"+-",round(np.std(list_dilate),3))
    print("Shape:", round(np.mean(list_shape),3),"+-",round(np.std(list_shape),3))
    print("Temporal:", round(np.mean(list_temporal),3),"+-",round(np.std(list_temporal),3))
    print("R2:", round(np.mean(list_r2),3),"+-",round(np.std(list_r2),3))
    print("--------------------------------------") 
    dict ={"RMSE":str(np.mean(list_rmse))+"+-"+str(np.std(list_rmse)),
           "Dilate":str(round(np.mean(list_dilate),3))+"+-"+str(round(np.std(list_dilate),3)),
           "Shape":str(round(np.mean(list_shape),3))+"+-"+str(round(np.std(list_shape),3)),
           "Temporal":str(round(np.mean(list_temporal),3))+"+-"+str(round(np.std(list_temporal),3)),
           "R2":str(round(np.mean(list_r2),3))+"+-"+str(round(np.std(list_r2),3))}

    pathfile = path + "metrics/validation.txt"
    with open(pathfile, 'w') as f:
        print(dict, file=f)

    list_rmse= np.array(list_rmse).reshape(-1,1)
    np.savetxt(path +"metrics/"+'rmse.txt', list_rmse, delimiter=',')

    list_dilate= np.array(list_dilate).reshape(-1,1)
    np.savetxt(path +"metrics/"+'dilate.txt', list_dilate, delimiter=',')

    list_shape= np.array(list_shape).reshape(-1,1)
    np.savetxt(path +"metrics/"+'shape.txt', list_shape, delimiter=',')

    list_temporal= np.array(list_temporal).reshape(-1,1)
    np.savetxt(path +"metrics/"+'temporal.txt', list_temporal, delimiter=',')

    list_r2= np.array(list_r2).reshape(-1,1)
    np.savetxt(path +"metrics/"+'r2.txt', list_r2, delimiter=',')

def rescale_data_eicu(np_data,seq_length,num_features,path):
    """ Function for rescaling data (eICU) """
    scaler=pickle.load(open(path+'data/'+'scaler_eicu','rb'))
    np_array_rescale= scaler.inverse_transform(np.reshape(np_data,(-1*seq_length,num_features)))
    np_array_rescale = np.around(np_array_rescale,2)
    np_array_rescale = np_array_rescale.reshape(-1,seq_length,num_features)
    return np_array_rescale

def splitting_data(x_data,database,seq_length,num_features,path):
    """ Function for splitting data (used for training a mortality prediction model) """
    x_data = np.array(x_data).reshape(-1*seq_length,num_features)
    path = 'experimentos/0cardioseries_rmse/'
    if database == "mimic":
        x_data_rescale = rescale_data(x_data,seq_length,num_features,path)
    else:
        x_data_rescale = rescale_data_eicu(real_target,seq_length,num_features,path)
    x_data_rescale = x_data_rescale.reshape(-1,seq_length*num_features)

    # ---- Getting label -----#
    data_pid = pd.read_csv("impute/data_ACS/test_acs_saits.csv")#Step 1: Get 'PID'
    demo_data = pd.read_csv("impute/data/mimic_demo_diagnoses.csv") #reading mortality label
    patients = set(data_pid.pid) #unique pid
    list_label=[]
    for pat in patients: #loop for extracting label
        ehr = demo_data[demo_data.PID == pat]
        list_label.append(ehr['MORTALITY_INUNIT'].item())
    np_y = np.array(list_label).reshape(-1) #convert list to np_array
    np_y = np_y[:x_data_rescale.shape[0]]
    xtrain, xtest, ytrain, ytest= train_test_split(x_data_rescale,np_y,test_size=0.2,random_state=1,stratify=np_y)  
    return xtrain, xtest, ytrain, ytest

def saving_model(model_ehr,model_name,logs_file):
    """ Function for saving mortality prediction model"""
    file_out = os.path.join(logs_file+str(model_name))
    with open(file_out, 'wb') as modelfile:
        pickle.dump(model_ehr,modelfile)

def xgb_model(xtrain, xtest, ytrain, ytest,filename,path):
    """ Function for training XGB model (TSTR, TRTS, TRTR)"""
    results = pd.DataFrame(columns=['accuracy','f1_score','auc_score'])
    sample_weights = class_weight.compute_sample_weight('balanced', ytrain)
    ratio = np.round(float(np.sum(ytrain == 0)) / np.sum(ytrain==1),3)
    estimator = xgb.XGBClassifier(objective = 'binary:logistic',seed = 422,scale_pos_weight = ratio,
                                 eval_metric= 'auc',learning_rate=0.003)
    model_xgb = estimator.fit(xtrain,ytrain)
    modelname = "xgb_"+str(filename)
    saving_model(model_xgb,modelname,path)
    
    # Metrics
    ypred = model_xgb.predict(xtest)
    acc = accuracy_score(ytest,ypred)  
    ap = average_precision_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, average='weighted')
    ypred_proba = model_xgb.predict_proba(xtest)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(ytest,ypred_proba)
    auc_score = auc(false_positive_rate, true_positive_rate)

    plot_roc_curve(false_positive_rate, true_positive_rate,auc_score,filename,path)
    plot_precision_recall_curve(ytest,ypred_proba,filename,path)## keep probabilities for the positive outcome only
    print("accuracy:",round(acc,3),"ap:",round(ap,3),"f1:",round(f1,3),"auc:",round(auc_score,3))
    values = {'accuracy':acc,'f1_score':f1,'auc_score':auc_score}
    results = results.append(values,ignore_index=True)
    results.to_csv(path +str(filename)+".csv")
    return model_xgb
    

def plotting_predictions(real_input,real_target,pred,mask,input_seq_length,output_seq_length,
                         num_features,num_conditions,conditions,path, net, num_samples,mode,fold):
    """ Function for analysing the time series predicted vs time series real"""
    feature_list = ['arterial_bp_mean','respiratory_rate', 'diastolic_bp','spo2','heart_rate', 'systolic_bp', 'temperature']
    
    real = np.array(real_input).reshape(-1*input_seq_length,num_features)
    target = np.array(real_target).reshape(-1*output_seq_length,num_features)
    prediction = np.array(pred).reshape(-1*output_seq_length,num_features)
    mask = np.array(mask).reshape(-1,output_seq_length,num_features)

    past_rescale = rescale_data(real,input_seq_length,num_features,path)
    expected_rescale = rescale_data(target,output_seq_length,num_features,path)
    pred_rescale = rescale_data(prediction,output_seq_length,num_features,path)

    past_rescale = np.array(past_rescale).reshape(-1,input_seq_length,num_features)
    expected_rescale = np.array(expected_rescale).reshape(-1,output_seq_length,num_features)
    pred_rescale = np.array(pred_rescale).reshape(-1,output_seq_length,num_features)

    np.random.seed(42) 
    indices = np.random.choice(range(past_rescale.shape[0]),replace=False,size=num_samples) #selection of indices
    for index in indices: 
        past_pat = past_rescale[index, :, :]
        expected_pat = expected_rescale[index,:, :]
        prediction_pat= pred_rescale[index,:,:]

        pd_past_pat = pd.DataFrame(past_pat,columns= feature_list)
        pd_expected_pat = pd.DataFrame(expected_pat,columns= feature_list)
        pd_prediction_pat = pd.DataFrame(prediction_pat,columns= feature_list)

        mask_pat = mask[index,:,:]
        mask_pat = mask_pat.flatten()
        expected_list = expected_pat.flatten()

        data_array = np.empty(expected_pat.shape[0]*expected_pat.shape[1])
        for i in range(len(mask_pat)):
            if mask_pat[i]==True:
                data_array[i] = expected_list[i]
            else:
                data_array[i] = np.nan

        data_array = data_array.reshape(expected_pat.shape[0],expected_pat.shape[1])
        pd_mask_pat = pd.DataFrame(data_array,columns = pd_expected_pat.columns)
        pd_mask_pat = rename_columns(pd_mask_pat)

        #Conditionig
        if num_conditions > 0:
            np_cond = np.array(conditions).reshape(-1,num_conditions)
            conditions_per_patient = np_cond[index]
            age = conditions_per_patient[0]
            sex = conditions_per_patient[1].astype(int)

            # ----Converting One-hot to text
            sex_data = {'Sex': [sex],}
            sex_pd = pd.DataFrame(sex_data)
            sex_pd['Sex'] = np.where(sex_pd['Sex'] == 1, 'Hombre','Mujer')
            #-------------#

        pd_past_pat = rename_columns(pd_past_pat)
        pd_expected_pat = rename_columns(pd_expected_pat)
        pd_prediction_pat = rename_columns(pd_prediction_pat)
        features_trans = pd_prediction_pat.columns

        fig = plt.figure(figsize=(12,10))
        plt.subplots_adjust(wspace=0.2,hspace=0.6)
        iplot = 420

        for i in range(7):
            feature = features_trans[i]
            iplot += 1
            if i == 6:
                ax = plt.subplot2grid((4,8), (i//2, 2), colspan=4)
            else:
                ax = plt.subplot2grid((4,2), (i//2,i%2))
            ax.plot(range(pd_past_pat.shape[0], pd_past_pat.shape[0]+pd_expected_pat.shape[0]+1),
                    np.concatenate([pd_past_pat[feature][pd_past_pat.shape[0]-1:pd_past_pat.shape[0]], pd_expected_pat[feature]]),
                    "r.-",label="Valor imputado",color = "orange")
        
            ax.plot(range(pd_past_pat.shape[0], pd_past_pat.shape[0]+pd_expected_pat.shape[0]+1)
                    ,np.concatenate([pd_past_pat[feature][pd_past_pat.shape[0]-1:pd_past_pat.shape[0]], pd_prediction_pat[feature]]),
                    'o--', label="Predicción",color='green',markersize=10,)

            ax.plot(range(pd_past_pat.shape[0], pd_past_pat.shape[0]+pd_expected_pat.shape[0]+1)
                    ,np.concatenate([pd_past_pat[feature][pd_past_pat.shape[0]-1:pd_past_pat.shape[0]],pd_mask_pat[feature]]),
                    marker='o',markersize=10,label="Valor real",color = "orange")

            ax.plot(range(1,pd_past_pat.shape[0]+1),pd_past_pat[feature],marker='o',markersize=10,label="Observaciones", color = "darkblue")

            ax.set_title(feature,fontsize=10)
            ax.grid(True,color='lightgrey',alpha=0.5)
            ax.set_xlabel("Tiempo (horas)",fontsize=10)
            ax.set_ylabel("Valores",fontsize=10)

            steps = np.arange(1, 13, 1)
            ax.set_xticks(steps)
            plt.autoscale(False)# Don't mess with the limits!

        if num_conditions == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),fancybox=True, shadow=True, ncol=5)

        if num_conditions == 5:
            dm = conditions_per_patient[2].astype(int)
            hyp = conditions_per_patient[3].astype(int)
            hd = conditions_per_patient[4].astype(int)

            ax.legend(title='Sexo: '+str(sex_pd['Sex'].item())+' Edad:'+str(age)
                      + ' Diabetes:'+str(dm) +' Hipertensión:'+str(hyp)+' Enfermedades cardiovasculares:'+str(hd),
                      loc='upper center', bbox_to_anchor=(0.5, -0.35),fancybox=True, shadow=True, ncol=5)
        plt.savefig(path+'plots/'+str(mode)+"_epoch_"+str(fold)+"_pat_"+str(index)+".png")
        plt.savefig(path+'plots/'+str(mode)+"_epoch_"+str(fold)+"_pat_"+str(index)+".pdf")
        plt.close()

def rename_columns(df):
    """ Function for renaming the features (from english to spanish )"""
    df = df[['systolic_bp', 'respiratory_rate','diastolic_bp', 'heart_rate','arterial_bp_mean','spo2','temperature']]
    df.rename(columns = {'systolic_bp':'Presión arterial sistólica (mmHg)',
                         'respiratory_rate':'Frecuencia respiratoria (bpm)',
                         'diastolic_bp':'Presión arterial diastólica (mmHg)',
                         'heart_rate':'Frecuencia cardiaca (bpm)',
                         'arterial_bp_mean':'Presión arterial media (mmHg)',
                         'spo2':'SpO2',
                         'temperature':'Temperatura (C)',
                         }, inplace = True)
    return df

def compute_confiance_interval(test_true,test_pred):
    """ Function for computing the confiance interval"""
    # estimate stdev of yhat
    sum_errs = arraysum((test_true - test_pred)**2)
    stdev = sqrt(1/(len(test_true)-2) * sum_errs)
    # calculate prediction interval
    interval = 1.96 * stdev

    print('Prediction Interval: %.3f' % interval)
    limite_inferior, limite_superior = yhat_out - interval, yhat_out + interval
    print('95%% likelihood that the true value is between %.3f and %.3f' % (lower, upper))
    print('True value: %.3f' % y_out)
    return limite_inferior, limite_superior


def dwprobability(test_true,test_pred,output_seq_length,num_features,path,net,fold,mode="train"):
    """Function for plotting the dimension wise probability graph"""
    test_true = np.array(test_true).reshape(-1,output_seq_length*num_features)
    test_pred = np.array(test_pred).reshape(-1,output_seq_length*num_features)

    prob_real = np.mean(test_true, axis=0).reshape(-1)
    prob_syn = np.mean(test_pred, axis=0).reshape(-1)

    p1 = plt.scatter(prob_real, prob_syn, c ='b', alpha=0.5)    
    x_max = max(np.max(prob_real), np.max(prob_syn))
    x_min = min(np.min(prob_real), np.min(prob_syn))
    x = np.linspace(x_min-0.5, x_max + 0.5)
    p2 = plt.plot(x, x, linestyle='--', color='gray', label="Ideal")  # solid
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.tick_params(labelsize=10)
    plt.legend(loc=2, prop={'size': 10})
    plt.title('Rendimiento de probabilidad por dimensión \n (datos reales vs datos predichos)')
    plt.xlabel('Datos reales')
    plt.ylabel('Predicción')
    plt.savefig(path +"plots/"+"dim_wise_proba_"+str(mode)+"_epoch_"+str(fold)+".png")
    plt.close()


def rescale_data_test(np_data,path):
    """Function for rescaling data"""
    scaler=pickle.load(open(path+'data/'+'scaler','rb'))
    np_data = np.array(np_data).reshape(-1*np_data.shape[1],np_data.shape[2])
    np_array_rescale= scaler.inverse_transform(np_data)
    np_array_rescale = np.around(np_array_rescale,3)
    return np_array_rescale

def computing_rmse(ytrue,ypred,output_seq_length,feature,mode,path):
    """Function for computing RMSE in all features (y_true vs y_pred)"""
    list_arterial,list_resp,list_dias,list_spo2,list_heart,list_sys,list_temp=[],[],[],[],[],[],[]

    print("*** Computing RMSE score ***")
    ytrue = np.array(ytrue).reshape(-1,output_seq_length,len(feature))
    ypred = np.array(ypred).reshape(-1,output_seq_length,len(feature))

    if mode =="escalado":
        ytrue = rescale_data_test(ytrue,path)#[32, 24, 8])
        ypred = rescale_data_test(ypred,path)#[32, 24, 8])
        ytrue = np.array(ytrue).reshape(-1,output_seq_length,len(feature))
        ypred = np.array(ypred).reshape(-1,output_seq_length,len(feature))

    for index in range(ytrue.shape[0]):
        ytrue_pat = ytrue[index, :, :]
        ypred_pat = ypred[index,:, :]
        ytrue_pd = pd.DataFrame(ytrue_pat,columns=feature)
        ypred_pd = pd.DataFrame(ypred_pat,columns=feature)
        list_arterial.append(math.sqrt(mean_squared_error(np.asarray(ytrue_pd.arterial_bp_mean),np.asarray(ypred_pd.arterial_bp_mean))))
        list_resp.append(math.sqrt(mean_squared_error(np.asarray(ytrue_pd.respiratory_rate),np.asarray(ypred_pd.respiratory_rate))))
        list_dias.append(math.sqrt(mean_squared_error(np.asarray(ytrue_pd.spo2),np.asarray(ypred_pd.spo2))))
        list_spo2.append(math.sqrt(mean_squared_error(np.asarray(ytrue_pd.spo2),np.asarray(ypred_pd.spo2))))
        list_heart.append(math.sqrt(mean_squared_error(np.asarray(ytrue_pd.heart_rate),np.asarray(ypred_pd.heart_rate))))
        list_sys.append(math.sqrt(mean_squared_error(np.asarray(ytrue_pd.systolic_bp),np.asarray(ypred_pd.systolic_bp))))
        list_temp.append(math.sqrt(mean_squared_error(np.asarray(ytrue_pd.temperature),np.asarray(ypred_pd.temperature))))

    print("* RMSE per feature")
    print("arterial_bp_mean:", len(list_arterial),round(np.mean(list_arterial),2),"+-",round(np.std(list_arterial),2))
    print("respiratory_rate:", len(list_resp),round(np.mean(list_resp),2),"+-",round(np.std(list_resp),2))
    print("diastolic_bp:", len(list_dias),round(np.mean(list_dias),2),"+-",round(np.std(list_dias),2))
    print("spo2:", len(list_spo2),round(np.mean(list_spo2),2),"+-",round(np.std(list_spo2),2))
    print("heart_rate:", len(list_heart),round(np.mean(list_heart),2),"+-",round(np.std(list_heart),2))
    print("systolic_bp:", len(list_sys),round(np.mean(list_sys),2),"+-",round(np.std(list_sys),2))
    print("temperature:", len(list_temp),round(np.mean(list_temp),2),"+-",round(np.std(list_temp),2))
    print("--------------------------------------")

    dict ={"arterial_bp_mean":str(round(np.mean(list_arterial),2))+"+-"+str(round(np.std(list_arterial),2)),
           "respiratory_rate":str(round(np.mean(list_resp),2))+"+-"+str(round(np.std(list_resp),2)),
           "diastolic_bp":str(round(np.mean(list_dias),2))+"+-"+str(round(np.std(list_dias),2)),
           "spo2":str(round(np.mean(list_spo2),2))+"+-"+str(round(np.std(list_spo2),2)),
           "heart_rate":str(round(np.mean(list_heart),2))+"+-"+str(round(np.std(list_heart),2)),
           "systolic_bp":str(round(np.mean(list_sys),2))+"+-"+str(round(np.std(list_sys),2)),
           "temperature":str(round(np.mean(list_temp),2))+"+-"+str(round(np.std(list_temp),2)),}

    pathfile = path + "metrics/rmse_per_feature_"+str(mode)+".txt"
    with open(pathfile, 'w') as f:
        print(dict, file=f)
    
def plotting_rmse(ytrue,ypred,output_seq_length,feature,mode,path):
    """Function for plotting the distribution for each feature (y_true vs y_pred). Included the confidence interval"""
    ytrue = np.array(ytrue).reshape(-1,output_seq_length,len(feature))
    ypred = np.array(ypred).reshape(-1,output_seq_length,len(feature))
    np_mean_ci, np_std_ci, np_qmin_ci, np_qmax_ci = confidence_interval(ytrue,ypred,feature,path)
    print("IC:(mean)",np_mean_ci)
    ytrue = rescale_data_test(ytrue,path)
    ypred = rescale_data_test(ypred,path)
    ytrue = np.array(ytrue).reshape(-1,output_seq_length,len(feature))
    ypred = np.array(ypred).reshape(-1,output_seq_length,len(feature))
    lista_arterial_real,lista_arterial_pred,lista_resp_real,lista_resp_pred,lista_diasbp_real,lista_diasbp_pred,lista_spo2_real,lista_spo2_pred, \
    lista_hr_real,lista_hr_pred, lista_sysbp_real,lista_sysbp_pred,lista_temp_real, lista_temp_pred = extraction_data_per_feature(ytrue,ypred,feature)

    df_features=pd.DataFrame({'arterial_real':lista_arterial_real,'arterial_pred':lista_arterial_pred,
                          'respiratory_rate_real':lista_resp_real,'respiratory_rate_pred':lista_resp_pred,
                          'diastolic_bp_real':lista_diasbp_real,'diastolic_bp_pred':lista_diasbp_real,
                          'spo2_real':lista_spo2_real,'spo2_pred':lista_spo2_pred,
                          'heart_rate_real':lista_hr_real,'heart_rate_pred':lista_hr_pred,
                          'systolic_bp_real':lista_sysbp_real,'systolic_bp_pred':lista_sysbp_pred,
                          'temperature_real':lista_temp_real,'temperature_pred':lista_temp_pred,})

    fig = plt.figure(figsize=(7,9)) #x,y
    plt.subplots_adjust(wspace=0.3,hspace=0.6)
    ax1 = plt.subplot(4, 2, 1)
    ax1.boxplot([df_features.arterial_real,df_features.arterial_pred], labels=["Real","Predicción"],showfliers=False)
    ax1.set_title('Presión arterial media \n (IC 95% '+str(np_mean_ci[0])+' ['+str(np_qmin_ci[0])+'-'+str(np_qmax_ci[0])+'])',fontsize=10)
    ax1.set_ylabel('mmHg')
    ax1.set_ylim([40,120])
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax2 = plt.subplot(4, 2, 2)
    ax2.boxplot([df_features.respiratory_rate_real,df_features.respiratory_rate_pred], labels=["Real","Predicción"],showfliers=False)
    ax2.set_title('Frecuencia respiratoria \n (IC 95% '+str(np_mean_ci[1])+' ['+str(np_qmin_ci[1])+'-'+str(np_qmax_ci[1])+'])',fontsize=10)
    ax2.set_ylabel('bpm')
    ax2.set_ylim([5,35])
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax3 = plt.subplot(4, 2, 3)
    ax3.boxplot([df_features.diastolic_bp_real,df_features.diastolic_bp_pred], labels=["Real","Predicción"],showfliers=False)
    ax3.set_title('Presión arterial diastólica \n (IC 95% '+str(np_mean_ci[2])+' ['+str(np_qmin_ci[2])+'-'+str(np_qmax_ci[2])+'])',fontsize=10)
    ax3.set_ylabel('mmHg')
    ax3.set_ylim([25,90])
    ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax4 = plt.subplot(4, 2, 4)
    ax4.boxplot([df_features.spo2_real,df_features.spo2_pred], labels=["Real","Predicción"],showfliers=False)
    ax4.set_title('SpO2 \n (IC 95% '+str(np_mean_ci[3])+' ['+str(np_qmin_ci[3])+'-'+str(np_qmax_ci[3])+'])',fontsize=10)
    ax4.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    ax4.set_ylim([92,101])

    ax5 = plt.subplot(4, 2, 5)
    ax5.boxplot([df_features.heart_rate_real,df_features.heart_rate_pred], labels=["Real","Predicción"],showfliers=False)
    ax5.set_title('Frecuencia cardiaca \n (IC 95% '+str(np_mean_ci[4])+' ['+str(np_qmin_ci[4])+'-'+str(np_qmax_ci[4])+'])',fontsize=10)
    ax5.set_ylabel('bpm')
    ax5.set_ylim([30,130])
    ax5.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax6 = plt.subplot(4, 2, 6)
    ax6.boxplot([df_features.systolic_bp_real,df_features.systolic_bp_pred], labels=["Real","Predicción"],showfliers=False)
    ax6.set_title('Presión arterial sistólica \n (IC 95% '+str(np_mean_ci[5])+' ['+str(np_qmin_ci[5])+'-'+str(np_qmax_ci[5])+'])',fontsize=10)
    ax6.set_ylabel('mmHg')
    ax6.set_ylim([60,165])
    ax6.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    ax7 = plt.subplot(4, 1, 4)
    ax7.boxplot([df_features.temperature_real,df_features.temperature_pred], labels=["Real","Predicción"],showfliers=False)
    ax7.set_title('Temperatura \n (IC 95% '+str(np_mean_ci[6])+' ['+str(np_qmin_ci[6])+'-'+str(np_qmax_ci[6])+'])', fontsize=10)
    ax7.set_ylabel('C')
    ax7.set_ylim([34,40])
    ax7.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

    plt.savefig(path+'plots/boxplot_per_feature_comparative.pdf')
    plt.close()

def extraction_data_per_feature(ytrue,ypred,feature):
    """ Function for extracting values for each feature (y_true vs y_pred)"""
    lst_arterial_real,lst_respiratory_rate_real,lst_diastolic_bp_real,lst_spo2_real,lst_heart_rate_real,lst_systolic_bp_real,lst_temperature_real=[],[],[],[],[],[],[]
    lst_arterial_pred,lst_respiratory_rate_pred,lst_diastolic_bp_pred,lst_spo2_pred,lst_heart_rate_pred,lst_systolic_bp_pred,lst_temperature_pred = [],[],[],[],[],[],[]

    for index in range(ytrue.shape[0]):
        ytrue_pat = ytrue[index, :, :]
        ypred_pat = ypred[index,:, :]

        ytrue_pd = pd.DataFrame(ytrue_pat,columns=feature)
        ypred_pd = pd.DataFrame(ypred_pat,columns=feature)
        for i in range(len(feature)):
            featurename = feature[i]
            if featurename == "arterial_bp_mean":
                lst_arterial_real.append(np.asarray(ytrue_pd[featurename]))
                lst_arterial_pred.append(np.asarray(ypred_pd[featurename]))
            if featurename == "respiratory_rate":
                lst_respiratory_rate_real.append(np.asarray(ytrue_pd[featurename]))
                lst_respiratory_rate_pred.append(np.asarray(ypred_pd[featurename]))
            if featurename == "diastolic_bp":
                lst_diastolic_bp_real.append(np.asarray(ytrue_pd[featurename]))
                lst_diastolic_bp_pred.append(np.asarray(ypred_pd[featurename]))
            if featurename == "spo2":
                lst_spo2_real.append(np.asarray(ytrue_pd[featurename]))
                lst_spo2_pred.append(np.asarray(ypred_pd[featurename]))
            if featurename == "heart_rate":
                lst_heart_rate_real.append(np.asarray(ytrue_pd[featurename]))
                lst_heart_rate_pred.append(np.asarray(ypred_pd[featurename]))
            if featurename == "systolic_bp":
                lst_systolic_bp_real.append(np.asarray(ytrue_pd[featurename]))
                lst_systolic_bp_pred.append(np.asarray(ypred_pd[featurename]))
            if featurename == "temperature":
                lst_temperature_real.append(np.asarray(ytrue_pd[featurename]))
                lst_temperature_pred.append(np.asarray(ypred_pd[featurename]))


    lst_arterial_real = np.array(lst_arterial_real).flatten()
    lst_arterial_pred = np.array(lst_arterial_pred).flatten()
    lst_respiratory_rate_real = np.array(lst_respiratory_rate_real).flatten()
    lst_respiratory_rate_pred = np.array(lst_respiratory_rate_pred).flatten()
    lst_diastolic_bp_real = np.array(lst_diastolic_bp_real).flatten()
    lst_diastolic_bp_pred = np.array(lst_diastolic_bp_pred).flatten()
    lst_spo2_real = np.array(lst_spo2_real).flatten()
    lst_spo2_pred = np.array(lst_spo2_pred).flatten()
    lst_heart_rate_real = np.array(lst_heart_rate_real).flatten()
    lst_heart_rate_pred = np.array(lst_heart_rate_pred).flatten()
    lst_systolic_bp_real = np.array(lst_systolic_bp_real).flatten()
    lst_systolic_bp_pred = np.array(lst_systolic_bp_pred).flatten()
    lst_temperature_real = np.array(lst_temperature_real).flatten()
    lst_temperature_pred = np.array(lst_temperature_pred).flatten()

    return lst_arterial_real,lst_arterial_pred,lst_respiratory_rate_real,lst_respiratory_rate_pred,\
        lst_diastolic_bp_real,lst_diastolic_bp_pred,lst_spo2_real,lst_spo2_pred, \
        lst_heart_rate_real,lst_heart_rate_pred, lst_systolic_bp_real,lst_systolic_bp_pred, \
        lst_temperature_real, lst_temperature_pred

def confidence_interval(ytrue,ypred,feature,path):
    """ Function to compute the confidence interval per feature"""
    from torchmetrics import MeanSquaredError, BootStrapper

    arterial_real,arterial_pred,resp_real,resp_pred,diasbp_real,diasbp_pred,spo2_real,spo2_pred, \
    hr_real,hr_pred, sysbp_real,sysbp_pred,temp_real, temp_pred = extraction_data_per_feature(ytrue,ypred,feature)

    mean_arterial, std_arterial, qmin_arterial, qmax_arterial = ci(arterial_real,arterial_pred)
    mean_resp, std_resp, qmin_resp, qmax_resp = ci(resp_real,resp_pred)
    mean_diasbp, std_diasbp, qmin_diasbp, qmax_diasbp = ci(diasbp_real,diasbp_pred)
    mean_spo2, std_spo2, qmin_spo2, qmax_spo2 = ci(spo2_real,spo2_pred)
    mean_hr, std_hr, qmin_hr, qmax_hr = ci(hr_real,hr_pred)
    mean_sysbp, std_sysbp, qmin_sysbp, qmax_sysbp = ci(sysbp_real,sysbp_pred)
    mean_temp, std_temp, qmin_temp, qmax_temp = ci(temp_real,temp_pred)

    #Creamos una lista con todos los resultados obtenidos de BootStrapper
    lista_mean_ci = [mean_arterial,mean_resp,mean_diasbp,mean_spo2,mean_hr,mean_sysbp,mean_temp]
    lista_std_ci = [std_arterial,std_resp,std_diasbp,std_spo2,std_hr,std_sysbp,std_temp]
    lista_qmin_ci = [qmin_arterial,qmin_resp,qmin_diasbp,qmin_spo2,qmin_hr,qmin_sysbp,qmin_temp]
    lista_qmax_ci = [qmax_arterial,qmax_resp,qmax_diasbp,qmax_spo2,qmax_hr,qmax_sysbp,qmax_temp]

    #From list to numpy
    np_mean_ci = np.array(lista_mean_ci).reshape(-1,len(lista_mean_ci))
    np_std_ci = np.array(lista_std_ci).reshape(-1,len(lista_std_ci))
    np_qmin_ci = np.array(lista_qmin_ci).reshape(-1,len(lista_qmin_ci))
    np_qmax_ci = np.array(lista_qmax_ci).reshape(-1,len(lista_qmax_ci))
    
    scaler=pickle.load(open(path+'data/'+'scaler','rb')) #loading scaler
    np_mean_ci = np.around(scaler.inverse_transform(np_mean_ci),2).flatten()
    np_std_ci = np.around(scaler.inverse_transform(np_std_ci),2).flatten()
    np_qmin_ci = np.around(scaler.inverse_transform(np_qmin_ci),2).flatten()
    np_qmax_ci = np.around(scaler.inverse_transform(np_qmax_ci),2).flatten()
    return np_mean_ci, np_std_ci, np_qmin_ci, np_qmax_ci


def ci(ytrue,ypred):
    """ Function for computing confidence interval using BootStrapper of pytorch""" 
    from torchmetrics import MeanSquaredError, BootStrapper
    torch.manual_seed(123)
    quantiles = torch.tensor([0.05, 0.95])
    base_metric = MeanSquaredError()
    bootstrap = BootStrapper(base_metric, num_bootstraps=10, sampling_strategy="multinomial", 
                            quantile=quantiles)
    bootstrap.update(torch.from_numpy(ypred), torch.from_numpy(ytrue))
    output = bootstrap.compute()

    resultados = list(output.values())
    mean = resultados[0].item()
    std = resultados[1].item()
    quantile = resultados[2]
    qmin = quantile[0]
    qmax = quantile[1]
    return mean, std, qmin,qmax

def dwprobability_con_metricas(test_true,test_pred,list_rmse,list_r2,output_seq_length,num_features,path,net,fold,mode="train"):
    """Function for plotting the dimension wise probability graph (testing)"""
    test_true = np.array(test_true).reshape(-1,output_seq_length*num_features)
    test_pred = np.array(test_pred).reshape(-1,output_seq_length*num_features)

    rmse_label = str(round(np.mean(list_rmse),3)) + '±' + str(round(np.std(list_rmse),3))
    r2_label = str(round(np.mean(list_r2),3)) + '±' + str(round(np.std(list_r2),3))
    prob_real = np.mean(test_true, axis=0).reshape(-1)
    prob_syn = np.mean(test_pred, axis=0).reshape(-1)

    #Plotting
    p1 = plt.scatter(prob_real, prob_syn, c ='b', alpha=0.5)    
    x_max = max(np.max(prob_real), np.max(prob_syn))
    x_min = min(np.min(prob_real), np.min(prob_syn))
    
    x = np.linspace(x_min-0.5, x_max + 0.5)
    p2 = plt.plot(x, x, linestyle='--', color='gray', label="Ideal")
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.text(0.7, 0.2, 'RMSE='+str(rmse_label))
    plt.text(0.7, 0.15, 'r2='+str(r2_label))
    plt.tick_params(labelsize=10)
    plt.legend(loc=2, prop={'size': 10})
    plt.xlabel('Series de tiempo reales')
    plt.ylabel('Series de tiempo predichas')
    plt.savefig(path +"plots/"+"dim_wise_proba_"+str(mode)+"_epoch_"+str(fold)+".pdf")
    plt.close()