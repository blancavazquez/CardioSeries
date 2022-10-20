## CardioSeries: a model for predicting multivariate time series

This repository contains the code  for predicting multivariate time series based on dilated causal convolutional neural networks (DCCNN). 
The application of this model is for early prediction of clinical behavior.

## Brief introduction
This repository is organized as follows:
* sql_scripts: contains a set of scripts to extract the EHR from MIMIC-III database.
* forecasting: presents the codes for evaluating of hyperparameters using 10 repetitions of 5-fold cross-validation. Included the scripts for loss functions and the DCCNN model. 
* analysis: consist of the codes for plotting all results and compared among them.
* imputation: describes the codes for evaluating several imputation methods.

## Requirements
For imputation, we used the libraries of:
- [Simple imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [Iterative imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)
- [Interpolate](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html)
- [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
- [Random](https://impyute.readthedocs.io/en/master/_modules/impyute/imputation/cs/random.html)
- [MICE](https://impyute.readthedocs.io/en/master/_modules/impyute/imputation/cs/mice.html)
- [Expectation - maximization](https://impyute.readthedocs.io/en/master/_modules/impyute/imputation/cs/em.html)
- [Moving window](https://impyute.readthedocs.io/en/master/_modules/impyute/imputation/ts/moving_window.html)
- [Self-Attention-based Imputation for Time Series](https://github.com/WenjieDu/SAITS)

You can create a conda environment with all the dependencies using the environment.yml file in this repository.
```
conda env create cardioseries.yml
```
## Citation:

```
B. Vazquez, “Modelos basados en aprendizaje de máquinas para el análisis de subpoblaciones de 
pacientes en registros clínicos electrónicos,” Ph.D. dissertation, IIMAS, UNAM, Mexico, 2022.
```
Thesis available at [URL](https://tesiunam.dgb.unam.mx/F/HBC87R6D53S5XAFSBKEAA7INY7SIHR4N3DGLCF4XCJK9LR76PU-26539?func=full-set-set&set_number=707136&set_entry=000006&format=999)
