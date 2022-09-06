#------
#super importante instalar: pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip
#------

import os
import datetime
import numpy as np
import pandas as pd

#Loading pytorch_lightning
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything #Trainer  ###comenté por estar repetida
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

#Loading models
from models.dcnn_causal import CausalModel

#Loading losses & utilities
from loss.losses import r2_corr, rmse_loss,mmd_loss,mae_loss
from loss.dilate_loss import dilate_loss
from utils import create_folders, loading_data,plotting_losses,extraction_demo_data

import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# ------- New libraries for cross-validation ------- #
from pytorch_lightning import LightningDataModule, seed_everything, Trainer, LightningModule
#from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.base import Loop

#from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import KFold, RepeatedKFold
from torch.utils.data.dataloader import DataLoader
import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Optional, Type
from torch.utils.data.dataset import Dataset, Subset
#-------------------------------#

start_time = datetime.datetime.now()
print("Start time:",start_time)

def split_df(df,in_seqlength,out_seqlength):
    """Split data: first hours (history), next hours (targets)"""
    start_index = 0
    end_index = in_seqlength+out_seqlength
    history = df[start_index:in_seqlength] #0-24 (first 24 hours)
    targets = df[in_seqlength:end_index] #(24-48) #next 24 hours
    return history, targets

def pad_arr(arr, expected_size):
    """
    Pad top of array when there is not enough history
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode='edge')
    return arr


def df_to_np(df,seq_length):
    """
    Convert dataframe to numpy
    """
    arr = np.array(df)
    arr = pad_arr(arr,seq_length)
    return arr


class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, target,demo, mask_saits,
                 input_seq_length, output_seq_length,num_features,path):
        self.groups = groups
        self.grp_by = grp_by
        self.features = target
        self.demo = demo
        self.mask_saits = mask_saits
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.num_features = num_features
        self.path = path

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        """ targets_in = first 24 hours, targets_out = next 24 hours """
        list_targets_in=[]
        pid_patient = self.groups[idx]

        df = self.grp_by.get_group(pid_patient) #get all features for each patient
        df_conditions = self.demo.get_group(pid_patient)
        df_conditions = df_conditions[['age','gender','diabetes','hypertension','heart_diseases']]

        #----- Get EHR real ------#
        #history=first24hrs, targets=next24hours
        history, targets = split_df(df,self.input_seq_length,self.output_seq_length) #history (24,10), targets (24,10)
        targets_in = history[self.features] #(24,8), first24hours
        targets_in = df_to_np(targets_in,self.input_seq_length)#(24,8)
        targets_in = targets_in.reshape(self.input_seq_length,self.num_features)
        targets_past = torch.tensor(targets_in, dtype=torch.float) #torch.Size([6, 8])

        targets_out = targets[self.features] #(24,8)#next24hours
        targets_out = np.array(targets_out) #(24,8)
        targets_out = targets_out.reshape(self.output_seq_length,self.num_features)
        targets_expected = torch.tensor(targets_out, dtype=torch.float) #torch.Size([6, 8])

        #Datos imputados con SAITS, incluye máscara
        df_mask = self.mask_saits.get_group(pid_patient)
        df_mask = df_mask[self.features]#[df_mask.columns[1:]]

        #"---Masking only for next 24 hours----"# 
        history_mask, targets_mask = split_df(df_mask,self.input_seq_length,self.output_seq_length)
        targets_mask = targets_mask.to_numpy()
        mask = torch.tensor(targets_mask,dtype=torch.float).eq(1).reshape(targets_mask.shape[0],targets_mask.shape[1]) #[6,8]

        # conditions
        conditions = np.around(np.array(df_conditions),1) #sex, age
        conditions = torch.tensor(conditions,dtype=torch.float) #[1,2]
        conditions = conditions.flatten()        
        return targets_past, targets_expected,conditions, mask

class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass

#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be splitted accordingly to        #
# the current fold split.                                                                   #
#############################################################################################

@dataclass
class KFoldDataModule(BaseKFoldDataModule):
    def __init__(self,data_train, data_val,train_mask,val_mask,
                feature_list,batch_size,
                input_seq_length,output_seq_length,
                num_features, path):
        super().__init__()
        self.data_train = data_train
        self.data_val = data_val
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.target_vars = feature_list
        self.batch_size = batch_size
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.num_features = num_features
        self.path = path

    def setup(self, stage: Optional[str] = None) -> None:
        seq_length = 6

        #ajustando datos
        train_np = self.data_train.to_numpy().reshape(-1,(seq_length*2)*self.data_train.shape[1]) #23520
        train_np = train_np[:30208].reshape(-1*(seq_length*2),self.data_train.shape[1]) #original 30,711, para 32=30688
        data_train = pd.DataFrame(train_np, columns = self.data_train.columns)
        
        val_np = self.data_val.to_numpy().reshape(-1,(seq_length*2)*self.data_val.shape[1]) #original 7703, para 32=7680
        val_np = val_np[:7680].reshape(-1*(seq_length*2),self.data_val.shape[1])
        data_val = pd.DataFrame(val_np, columns = self.data_val.columns)

        train_demo, val_demo = extraction_demo_data()
        grp_by_train = data_train.groupby(by=['pid'])
        grp_by_val = data_val.groupby(by=['pid'])
        groups_train = list(grp_by_train.groups)
        groups_val= list(grp_by_val.groups)

        grp_by_train_mask = train_mask.groupby(by=['pid'])
        grp_by_val_mask = val_mask.groupby(by=['pid'])
            
        #demographic data
        grp_by_train_demo = train_demo.groupby(by=['PID'])
        grp_by_val_demo = val_demo.groupby(by=['PID'])

        full_groups_train = [grp for grp in groups_train if grp_by_train.get_group(grp).shape[0]>=2*seq_length]
        full_groups_val = [grp for grp in groups_val if grp_by_val.get_group(grp).shape[0]>=2*seq_length]

        self.train_dataset = Dataset(groups=full_groups_train,grp_by=grp_by_train,
                        target=self.target_vars,#son todas las vars, menos pids y offset
                        demo = grp_by_train_demo, #sex,gender
                        mask_saits = grp_by_train_mask,
                        input_seq_length = self.input_seq_length,
                        output_seq_length = self.output_seq_length,
                        num_features = self.num_features,path = self.path)

        self.test_dataset = Dataset(groups=full_groups_val,grp_by=grp_by_val,
                       target=self.target_vars,#son todas las vars, menos pids y offset
                       demo = grp_by_val_demo,#sex,gender
                       mask_saits = grp_by_val_mask,
                       input_seq_length = self.input_seq_length,
                       output_seq_length = self.output_seq_length,
                       num_features = self.num_features,path = self.path)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        rkf = RepeatedKFold(n_splits=num_folds, n_repeats=10)
        self.splits = [split for split in rkf.split(range(len(self.train_dataset)))]
        #self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))] #original

    def setup_fold_index(self, fold_index: int) -> None:
        """`setup_fold_index`, the provided train dataset will be splitted accordingly to the current fold split."""
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)
        print("Train (fold): ", fold_index + 1, "(indices)",len(self.train_fold))
        print("Valid (fold): ", fold_index + 1, "(indices)",len(self.val_fold))

    def train_dataloader(self) -> DataLoader:
        #Dataloader  = #num_samples/batch_size
        train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=6,shuffle=True)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=5,shuffle=False)
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=5,shuffle=False)
        return test_loader

#############################################################################################
#                           Step 3 / 5: Implement the EnsembleVotingModel module            #
# The `EnsembleVotingModel` will take our custom LightningModule and                        #
# several checkpoint_paths.                                                                 #
#                                                                                           #
#############################################################################################
from torchmetrics.functional import accuracy

class EnsembleVotingModel(LightningModule):
    def __init__(self, path, batch_size,output_seq_length,num_features,alpha,gamma,
                model_cls: Type[LightningModule], checkpoint_paths: List[str]):
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.path = path
        self.batch_size = batch_size
        self.output_seq_length = output_seq_length
        self.num_features = num_features
        self.alpha = alpha
        self.gamma = gamma

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        num_conditions = 5
        target_in, target_out,condition, mask = batch
        condition = condition[:,:num_conditions]
        target_in = torch.tensor(target_in, dtype=torch.float32).to(target_in.device)
        target_out = torch.tensor(target_out, dtype=torch.float32).to(target_out.device)
        
        logits = torch.stack([m(target_in,condition) for m in self.models]).mean(0)
        logits = torch.masked_select(logits, mask)
        target = torch.masked_select(target_out, mask)
        
        #rmse = rmse_loss(logits, target)
        #mae = mae_loss(logits,target)
        #mmd = mmd_loss(logits,target)

        dilate,loss_shape,loss_temporal = dilate_loss(logits,target,batch_size=self.batch_size,
                                 seq_length = self.output_seq_length,num_features=self.num_features,
                                 alpha=self.alpha,gamma=self.gamma,device=target_in.device,
                                 mask = "True")
        self.log('last_loss', loss_temporal) 


#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several steps    #
# to do so which are described in detail within the documentation                           #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html.                 #
# Here, we will implement an outer fit_loop. It means we will implement subclass the        #
# base Loop and wrap the current trainer `fit_loop`.                                        #
#############################################################################################

class KFoldLoop(Loop):
    def __init__(self, num_folds: int, batch_size: int,
                input_seq_length: int, output_seq_length: int,
                num_features: int, alpha: float,gamma: float,
                export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.num_features = num_features
        self.alpha = alpha
        self.gamma = gamma
        self.current_fold: int = 0
        self.export_path = export_path+'models/'
        self.path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        """ Connects a training epoch loop to this fit loop. """
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"\nSTARTING FOLD {self.current_fold+1}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()
        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.self.current_fold :

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        #self.replace(fit_loop=FitLoop)
        self.replace(fit_loop=FitLoop(self.fit_loop.min_epochs, self.fit_loop.max_epochs))

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(self.path,self.batch_size,self.output_seq_length,
                                           self.num_features,self.alpha,self.gamma,
                                           type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

if __name__ == "__main__":
    import argparse
    seed_everything(42) #for reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--foldername")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decay",type=float,default=1e-3)
    parser.add_argument("--net", type=str)
    parser.add_argument("--alpha",type=float,default=1e-2)
    parser.add_argument("--gamma",type=float,default=1e-2)
    parser.add_argument("--kfold",type=int,default=5)
    parser.add_argument("--input_seq_length",type=int,default=12)
    parser.add_argument("--output_seq_length",type=int,default=12)
    parser.add_argument("--output_rnn",type=int,default=7)
    parser.add_argument("--num_features",type=int,default=8)
    parser.add_argument("--conditions",type=int,default=2)
    parser.add_argument("--missing_values",default="False")
    parser.add_argument("--saits_impute",default="False")
    parser.add_argument("--feature_list",type=str,default="temperature")
    args = parser.parse_args()
    feature_list = list(args.feature_list.split(" "))

    print("*************************************************************")
    print("*************************************************************")
    path = create_folders(args.foldername, args.input_seq_length, args.output_seq_length,args.conditions)
    print("*** Saving results in: ", path, "***")

    print("Epochs:",args.epochs,"Batch_size:",args.batch_size, "conditions:", args.conditions, "Out_seqlength:",args.output_seq_length)

    print("*** Loading data ***")
    data_train, data_val, train_mask,val_mask = loading_data(feature_list,path)
    
    model = CausalModel(w_decay=args.decay,dropout=0.1,alpha=args.alpha,
                        gamma=args.gamma,input_seq_length=args.input_seq_length,
                        output_seq_length=args.output_seq_length,output_rnn = args.output_rnn, 
                        num_features=args.num_features,batch_size=args.batch_size,lr = args.learning_rate,
                        num_conditions=args.conditions,path = path,feature_list= feature_list,
                        net = args.net,in_channels=32,out_channels=2,
                        kernel_size=1,stride=1,dilation=1,groups=1,bias=False).to(device)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',mode="min",
                                          dirpath=path+str('models/'),
                                          filename=str(args.net)+'_{epoch:02d}-{val_loss:.4f}',)
    
    logger = TensorBoardLogger(save_dir=path+'/logs', version=1)

    datamodule = KFoldDataModule(data_train, data_val,train_mask,val_mask,
                                feature_list,args.batch_size,
                                args.input_seq_length,args.output_seq_length,
                                args.num_features, path)

    trainer = Trainer(max_epochs=args.epochs,
                      gpus=1 if torch.cuda.is_available() else 0,
                      logger = False,
                      num_sanity_val_steps=0,
                      accelerator="auto",
                      enable_model_summary=True,
                      callbacks=[checkpoint_callback])


    print("*** Starting training:",args.net,"***")
    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(args.kfold, args.batch_size, args.input_seq_length,
                                args.output_seq_length,args.num_features,
                                args.alpha, args.gamma, export_path=path)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule=datamodule)
    #trainer.test(model, datamodule=datamodule)

    print(" *** Hparams ***")
    print(model.hparams)

    print("*** Plotting losses ***")
    plotting_losses(path,args.net)

    print("*** Saving parameters ***")
    dict_parameters = {'foldername':path,'epochs':args.epochs,'batch_size':args.batch_size,
                       'learning_rate':args.learning_rate,'decay':args.decay,'net':args.net,
                       'alpha':args.alpha,'gamma':args.gamma,'kfold':args.kfold,
                       'input_seq_length':args.input_seq_length,
                       'output_seq_length':args.output_seq_length,
                       'output_rnn':args.output_rnn,'num_features':args.num_features,
                       'conditions':args.conditions,'missing_values':args.missing_values,
                       'saits_impute':args.saits_impute,'feature_list':args.feature_list}
    with open(path+"models/parameters.txt", 'w') as f: 
        for key, value in dict_parameters.items(): 
            f.write('%s:%s\n' % (key, value))

print("End time:", datetime.datetime.now() - start_time, "Go to folder:", args.foldername)
