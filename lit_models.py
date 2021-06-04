
import logging
from typing import Optional

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import TripletMarginLoss

from lit_template import LitSystemTemplate
from lightly.loss import SymNegCosineSimilarityLoss

class LitSelfModel(LitSystemTemplate):
    def __init__(self,
                 model,
                 criterion:torch.nn,
                 optim:str,
                 lr:float,
                 epoch:int,
                 steps_per_epoch:int, #len(train_loader)
                  ):
        
        super().__init__(
                         lr,
                         optim,
                         epoch=epoch,
                         steps_per_epoch=steps_per_epoch)
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        
                
        self.criterion=criterion
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

        self.log('train_loss_ssl', loss)
        
        return loss
    
class LitClassifier(LitSystemTemplate):
    def __init__(self, model:LitSystemTemplate,optim:str,num_class:int=196):
        super().__init__(
            optim=optim,
            epoch=self.model.epoch,
            lr=self.model.lr,
            steps_per_epoch=self.model.steps_per_epoch
        )
        # create a moco based on ResNet
        self.model = model

        # freeze the layers of moco
        for p in self.model.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(512, num_class)  #hacer el 512 variable pero más adelante
        
    def forward(self, x):
        with torch.no_grad():
            y_hat = self.model.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        self.calculate_metrics(y_hat,y,self.valid_metrics_base)