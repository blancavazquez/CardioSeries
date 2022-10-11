"""
Code for mortality prediction (TSTR,TRTS,TRTR)
python3 forecasting/mortality.py
Author: Blanca Vázquez
"""
import shap
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import splitting_data, xgb_model, saving_predictions

##------------------------##
## Settings
database = "mimic"
input_seq_length = 6
output_seq_length = 6
features = ['Presión arterial media (mmHg)','Frecuencia respiratoria (bpm)', 'Presión arterial diastólica (mmHg)',
			'SpO2','Frecuencia cardiaca (bpm)', 'Presión arterial sistólica (mmHg)', 'Temperatura (C)']


num_features = len(features)
path = "experiments/dilate/metrics_stemi/"
##------------------------##

def summary_plot_bar(xgb_model, xtrain,xtest,feature_list):
	data = np.concatenate((xtrain, xtest), axis=0)
	explainer = shap.TreeExplainer(xgb_model)
	shap_values = explainer.shap_values(data)

	###reshape
	data= data.reshape(-1,len(feature_list))
	shap_values= shap_values.reshape(-1,len(feature_list))

	fig = plt.figure(figsize=(15,10))
	shap.summary_plot(shap_values,data,feature_names=feature_list,plot_type="bar")
	plt.show()

print("* Opening files:")
yreal = np.loadtxt(path +'yreal.txt', delimiter=',')
ypred = np.loadtxt(path +'ypred.txt', delimiter=',')
print("yreal:", len(yreal),"ypred:", len(ypred))


print("* Split train & test sets")
xtrain_real, xtest_real, ytrain_real, ytest_real = splitting_data(yreal,database,input_seq_length,num_features,path)
xtrain_synth, xtest_synth, ytrain_synth, ytest_synth = splitting_data(ypred,database, output_seq_length,num_features,path)

print("***** Train real - test real *****")
model_trtr = xgb_model(xtrain_real, xtest_real, ytrain_real, ytest_real,"trtr",path)
summary_plot_bar(model_trtr,xtrain_real,xtest_real,features)

print("**** Train synth - test synth ****")
model_tsts = xgb_model(xtrain_synth, xtest_synth, ytrain_synth, ytest_synth,"tsts",path)
summary_plot_bar(model_tsts,xtrain_synth,xtest_synth,features)

print("**** Train real - test synth ****")
model_trts = xgb_model(xtrain_real, xtest_synth, ytrain_real, ytest_synth,"trts",path)
summary_plot_bar(model_trts,xtrain_real,xtest_synth,features)

print("**** Train synth - test real ****")
model_tstr = xgb_model(xtrain_synth, xtest_real, ytrain_synth, ytest_real,"tstr",path)
summary_plot_bar(model_tstr,xtrain_synth,xtest_real,features)