
from typing import Optional
from lightly import utils

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils import data

from config import Optim
from metrics import get_metrics_collections_base

import logging
from lightly.utils import BenchmarkModule
from pytorch_lightning import LightningModule
class LitSystemTemplate(LightningModule):
    def __init__(self,
                 
                  lr:float=0.01,
                  optim_name:str="adam",
                  epoch:Optional[int]=None,
                  ):
        
        LightningModule.__init__(self)



        self.valid_metrics_base=get_metrics_collections_base(prefix="valid_")

        # log hyperparameters
        # self.save_hyperparameters()    
        self.lr=lr
        self.epochs=epoch
        self.optim_name=optim_name
        self.optim=Optim[optim_name.lower()]
        
    
    def on_epoch_start(self):
        # torch.cuda.empty_cache()
        pass
            
    def configure_optimizers(self):
        if self.optim==Optim.adam:
            optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim==Optim.sgd:
            optimizer= torch.optim.SGD(self.parameters(), lr=self.lr,momentum=0.9,weight_decay=5e-4)
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]

    def insert_each_metric_value_into_dict(self,data_dict:dict,prefix:str):
 
        on_step=False
        on_epoch=True 
        
        for metric,value in data_dict.items():
            if metric != "preds":
                
                self.log("_".join([prefix,metric]),value,
                        on_step=on_step, 
                        on_epoch=on_epoch, 
                        logger=True
                )
                
    def add_prefix_into_dict_only_loss(self,data_dict:dict,prefix:str=""):
        data_dict_aux={}
        for k,v in data_dict.items():            
            data_dict_aux["_".join([prefix,k])]=v
            
        return data_dict_aux
    
    def calculate_metrics(self,y_hat,target,split_metric,):

        preds_probability=y_hat.softmax(dim=1)
            
        try:
            data_dict=split_metric(preds_probability,target)
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
    
        except Exception as e:
            print(e)
            sum_by_batch=torch.sum(preds_probability,dim=1)
            logging.error("la suma de las predicciones da distintos de 1, procedemos a imprimir el primer elemento")
            print(sum_by_batch)
            print(e)
            
    def calculate_loss_total(self,loss:dict,split:str):
        loss_total=sum(loss.values())
        data_dict={
                    f"{split}_loss_total":loss_total,
                    **loss,
                    }
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        return loss_total

class LitSelfModel(LitSystemTemplate,BenchmarkModule):
    def __init__(self,
                 model,
                 criterion:torch.nn,
                 optim:str,
                 lr:float,
                 epoch:int,
                 num_ftrs:int,
                 dataloader_kNN:torch.utils.data.DataLoader,
                 num_classes:int
                  ):
        
        LitSystemTemplate.__init__(self,
                         lr,
                         optim,
                         epoch=epoch,
        )
        BenchmarkModule.__init__(self,
            dataloader_kNN=dataloader_kNN,
            num_classes=num_classes,
        )
        
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        self.num_ftrs=num_ftrs
                
        self.criterion=criterion
        self.backbone=model.backbone
        self.model=model     
        
    def forward(self,x):
        self.model(x)        
    
    def training_step(self, batch, batch_idx):
        # get the two image transformations
        (x0, x1), _, _ = batch
        # forward pass of the transformations

        y0, y1 = self.model(x0, x1)
        # calculate loss

        loss = self.criterion(y0, y1)
        # log loss and return

        self.log('train_loss_ssl', loss,prog_bar=True)
        
        return loss
    
    def on_epoch_end(self) -> None:
        self.log("KNN_accuracy_max",self.max_accuracy*100)
        return super().on_epoch_end()