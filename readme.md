## CardioSeries: a model for predicting multivariate time series

This repository contains the code for early prediction of clinical behavior based on dilated causal convolutional neural networks (DCCNN). 

## Brief introduction
This repository is organized as follows:
* sql_scripts: contains a set of scripts to extract the EHR from MIMIC-III database.
* forecasting: presents the codes for evaluating of hyperparameters using 10 repetitions of 5-fold cross-validation. Included the scripts for loss functions and the DCCNN model. 
* analysis: consist of the codes for plotting all results and compared among them.
* imputation: describes the codes for evaluating several imputation methods.

## Requirements
For imputation, we used the libraries of:
- [Simple imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
