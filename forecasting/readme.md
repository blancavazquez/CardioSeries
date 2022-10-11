## Training models

Here are the required steps to train the mdels. It assumes that you already have MIMIC-III dataset and the timeseries.

1) For training with kfolds:
```
bash kfold_training.sh 
```

2) For training after hyperparameter selection:
```
bash training.sh 
```

3) For testing:
```
bash testing.sh 
```
4) For training the mortality prediction models on sets: TrainReal_TestReal, TrainSyn_TestReal, TrainReal_TestSyn. This command is excuted after of testing.

python3 mortality.py
```
