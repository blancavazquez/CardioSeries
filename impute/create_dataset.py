""""
Script for selecting clinical data (without missing data)
input: real_validation_set (with missing data)
output: dataframe without missing data
Author: Blanca Vázquez
"""

import pandas as pd
import numpy as np

path_data = "../impute/"
data=pd.read_csv(path_data+"val_real.csv")
print("Real size:", data.shape)

newdf = data[["heart_rate","systolic_bp","diastolic_bp","respiratory_rate","temperature","spo2","arterial_bp_mean"]]
print("Subset:", newdf.shape)

newdf = newdf.dropna().reset_index()
newdf = newdf.drop(["index"],axis=1)

#Saving data
newdf.to_csv(path_data+"newDF.csv")