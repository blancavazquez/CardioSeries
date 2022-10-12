"""
Code for evaluating several imputation methods
Author: Blanca Vázquez
"""

import os
import numpy as np
import math
import numpy.ma as ma #Masking!!
import pandas as pd
import time
import impyute as impy
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
np.random.seed(242)

"""" Load data """
path_data = "data/"
df_original = pd.read_csv(path_data+"original.csv")
df_original = df_original.astype(float) # convert all to float

""" Dataframe para guardar resultados """
resultados = pd.DataFrame(columns=['metodo','porcentaje','tiempo','num_repeticion'])
scaler = MinMaxScaler(feature_range=(0,1)).fit(df_original)

def rmse(ytrue,ypred,mask):
    mask = ma.make_mask(mask.to_numpy().reshape(-1,1))
    mse = mean_squared_error(ytrue[mask],ypred[mask])
    return math.sqrt(mse)

def simple_imputation(data,resultados,porcentaje,data_original,mask, num_rep):
    imputation = "simple"
    print("* Starting "+ str(imputation)+ " imputation")
    start = time.time()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(data.to_numpy())
    data_imputed = imputer.transform(data.to_numpy())
    runtime = time.time() - start
    rmse_score = rmse(data_original.to_numpy().reshape(-1,1),data_imputed.reshape(-1,1),mask)
    unscaled = pd.DataFrame(np.around(scaler.inverse_transform(data_imputed),2),columns = data.columns) #
    unscaled.to_csv("data/imp_"+str(imputation)+"_porcentaje_"+str(porcentaje)+".csv") #saving imputed data
    values = {'metodo':str(imputation),'porcentaje':str(porcentaje), 'tiempo':runtime,'rmse':rmse_score,'num_repeticion':num_rep}
    resultados = resultados.append(values,ignore_index=True) #saving resultados
    return resultados

def iterative_imputation(data,resultados,porcentaje,data_original,mask,num_rep):
    imputation = "iterative"
    print("* Starting "+ str(imputation)+ " imputation")
    start = time.time()
    imputer = IterativeImputer(imputation_order = 'ascending',random_state=242,min_value=-1,max_value=1).fit(data.to_numpy())
    data_imputed = imputer.transform(data.to_numpy())
    runtime = time.time() - start
    rmse_score = rmse(data_original.to_numpy().reshape(-1,1),data_imputed.reshape(-1,1),mask)
    unscaled = pd.DataFrame(np.around(scaler.inverse_transform(data_imputed),2),columns = data.columns)
    unscaled.to_csv("data/imp_"+str(imputation)+"_porcentaje_"+str(porcentaje)+".csv") #saving imputed data
    values = {'metodo':str(imputation),'porcentaje':str(porcentaje), 'tiempo':runtime,'rmse':rmse_score,'num_repeticion':num_rep}
    resultados = resultados.append(values,ignore_index=True) #saving resultados
    return resultados

def interpolation_linear_imputation(data,resultados,porcentaje,data_original,mask,num_rep):
    imputation = "interpolate_linear"
    print("* Starting "+ str(imputation)+ " imputation")
    start = time.time()
    data_imputed = data.interpolate(method='linear',limit_direction='both',axis=0).to_numpy()
    runtime = time.time() - start
    rmse_score = rmse(data_original.to_numpy().reshape(-1,1),data_imputed.reshape(-1,1),mask)
    unscaled = pd.DataFrame(np.around(scaler.inverse_transform(data_imputed),2),columns = data.columns)
    unscaled.to_csv("data/imp_"+str(imputation)+"_porcentaje_"+str(porcentaje)+".csv") #saving imputed data
    values = {'metodo':str(imputation),'porcentaje':str(porcentaje), 'tiempo':runtime,'rmse':rmse_score,'num_repeticion':num_rep}
    resultados = resultados.append(values,ignore_index=True) #saving resultados
    return resultados

def knn_imputation(data,resultados,porcentaje,data_original,mask,num_rep):
    imputation = "knn"
    print("* Starting "+ str(imputation)+ " imputation")
    start = time.time()
    imputer = KNNImputer(n_neighbors=5, weights ="distance").fit(data.to_numpy())
    data_imputed = imputer.transform(data.to_numpy()) #impute data
    runtime = time.time() - start
    rmse_score = rmse(data_original.to_numpy().reshape(-1,1),data_imputed.reshape(-1,1),mask)
    unscaled = pd.DataFrame(np.around(scaler.inverse_transform(data_imputed),2),columns = data.columns)
    unscaled.to_csv("data/imp_"+str(imputation)+"_porcentaje_"+str(porcentaje)+".csv") #saving imputed data
    values = {'metodo':str(imputation),'porcentaje':str(porcentaje), 'tiempo':runtime,'rmse':rmse_score,'num_repeticion':num_rep}
    resultados = resultados.append(values,ignore_index=True) #saving resultados
    return resultados

def moving_window_imputation(data,resultados,porcentaje,data_original,mask,num_rep):
    imputation = "moving_window"
    print("* Starting "+ str(imputation)+ " imputation")
    start = time.time()
    data_imputed = impy.moving_window(data.to_numpy())
    data_imputed = np.nan_to_num(data_imputed, nan=0)
    runtime = time.time() - start
    rmse_score = rmse(data_original.to_numpy().reshape(-1,1),data_imputed.reshape(-1,1),mask)
    unscaled = pd.DataFrame(np.around(scaler.inverse_transform(data_imputed),2),columns = data.columns)
    unscaled.to_csv("data/imp_"+str(imputation)+"_porcentaje_"+str(porcentaje)+".csv") #saving imputed data
    values = {'metodo':str(imputation),'porcentaje':str(porcentaje), 'tiempo':runtime,'rmse':rmse_score,'num_repeticion':num_rep}
    resultados = resultados.append(values,ignore_index=True) #saving resultados
    return resultados

def random_imputation(data,resultados,porcentaje,data_original,mask,num_rep):
    imputation = "random"
    print("* Starting "+ str(imputation)+ " imputation")
    start = time.time()
    data_imputed = impy.random(data.to_numpy())
    runtime = time.time() - start
    rmse_score = rmse(data_original.to_numpy().reshape(-1,1),data_imputed.reshape(-1,1),mask)
    unscaled = pd.DataFrame(np.around(scaler.inverse_transform(data_imputed),2),columns = data.columns)
    unscaled.to_csv("data/imp_"+str(imputation)+"_porcentaje_"+str(porcentaje)+".csv") #saving imputed data
    values = {'metodo':str(imputation),'porcentaje':str(porcentaje), 'tiempo':runtime,'rmse':rmse_score,'num_repeticion':num_rep}
    resultados = resultados.append(values,ignore_index=True) #saving resultados
    return resultados

def mice_imputation(data,resultados,porcentaje,data_original,mask,num_rep):
    imputation = "mice"
    print("* Starting "+ str(imputation)+ " imputation")
    start = time.time()
    data_imputed = impy.mice(data.to_numpy())
    runtime = time.time() - start
    rmse_score = rmse(data_original.to_numpy().reshape(-1,1),data_imputed.reshape(-1,1),mask)
    unscaled = pd.DataFrame(np.around(scaler.inverse_transform(data_imputed),2),columns = data.columns)
    unscaled.to_csv("data/imp_"+str(imputation)+"_porcentaje_"+str(porcentaje)+".csv") #saving imputed data
    values = {'metodo':str(imputation),'porcentaje':str(porcentaje), 'tiempo':runtime,'rmse':rmse_score,'num_repeticion':num_rep}
    resultados = resultados.append(values,ignore_index=True) #saving resultados
    return resultados

def expectation_maximization_imputation(data,resultados,porcentaje,data_original,mask,num_rep):
    imputation = "expectation_maximization"
    print("* Starting "+ str(imputation)+ " imputation")
    start = time.time()
    data_imputed = impy.em(data.to_numpy(),loops=5)
    runtime = time.time() - start
    rmse_score = rmse(data_original.to_numpy().reshape(-1,1),data_imputed.reshape(-1,1),mask)
    unscaled = pd.DataFrame(np.around(scaler.inverse_transform(data_imputed),2),columns = data.columns)
    unscaled.to_csv("data/imp_"+str(imputation)+"_porcentaje_"+str(porcentaje)+".csv") #saving imputed data
    values = {'metodo':str(imputation),'porcentaje':str(porcentaje), 'tiempo':runtime,'rmse':rmse_score,'num_repeticion':num_rep}
    resultados = resultados.append(values,ignore_index=True) #saving resultados
    return resultados

##############################################
num_rep = 1
for num_rep in range(10):
    print("* No. rep: ", num_rep)
    for percent_miss in [10,20,30,40,50]: #
        print("\n\n===> Porcentaje de datos a imputar: ", percent_miss, "%")
        columns = df_original.columns
        df_to_impute = df_original.copy()
        df_mask = pd.DataFrame(np.zeros((df_to_impute.shape[0], df_to_impute.shape[1])),columns=df_original.columns)
        n_samples = round(len(df_original)*percent_miss/100) # number of samples to make missing
        for col in columns:
            index = df_original[col].loc[df_original[col].notna()].sample(n_samples).index# select indices to replace with nan
            df_to_impute.loc[index, col] = np.nan
            df_mask.loc[index,col] = 1

        df_miss = df_to_impute.copy()
        df_miss.to_csv("data/datos_imputados_"+str(percent_miss)+".csv")
        df_pred = pd.DataFrame(scaler.transform(df_miss), columns=df_original.columns) #Scaling data (ypred)
        df_true = pd.DataFrame(scaler.transform(df_original), columns=df_original.columns) #Scaling data (ypred)
        resultados = simple_imputation(df_pred,resultados,percent_miss,df_true,df_mask,num_rep)
        resultados = iterative_imputation(df_pred,resultados,percent_miss,df_true,df_mask,num_rep)
        resultados = interpolation_linear_imputation(df_pred,resultados,percent_miss,df_true,df_mask,num_rep)
        resultados = knn_imputation(df_pred,resultados,percent_miss,df_true,df_mask,num_rep)
        resultados = random_imputation(df_pred,resultados,percent_miss,df_true,df_mask,num_rep)
        resultados = mice_imputation(df_pred,resultados,percent_miss,df_true,df_mask,num_rep)
        resultados = expectation_maximization_imputation(df_pred,resultados,percent_miss,df_true,df_mask,num_rep)
        resultados = moving_window_imputation(df_pred,resultados,percent_miss,df_true,df_mask,num_rep)
resultados.to_csv("data/resultados.csv") # Saving results