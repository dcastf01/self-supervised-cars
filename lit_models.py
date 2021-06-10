
import logging
from typing import Optional
from lightly.utils.benchmarking import BenchmarkModule

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F

from lit_template import LitSystemTemplate,LitSelfModel
import lightly

class LitNNCLRSelf(LitSelfModel):
    def __init__(self,
                backbone:nn.modules,
                 optim: str,
                 lr: float,
                 epoch: int,
                 num_ftrs: int,
                 dataloader_kNN: torch.utils.data.DataLoader,
                 num_classes: int
                 ):
                
        # esta pendiente de revisión ya que se necesita lo del banco de memoria o algo así
        criterion=lightly.loss.NTXentLoss()
        
        model=lightly.models.NNCLR(backbone,num_ftrs=num_ftrs)
        
        super().__init__(model,
                         criterion,
                         optim,
                         lr,
                         epoch,
                         num_ftrs,
                         dataloader_kNN,
                         num_classes
                         )
class LitBarlowTwinsSelf(LitSelfModel):
    def __init__(self,
                backbone:nn.modules,
                 optim: str,
                 lr: float,
                 epoch: int,
                 num_ftrs: int,
                 dataloader_kNN: torch.utils.data.DataLoader,
                 num_classes: int
                 ):
                
        criterion=lightly.loss.BarlowTwinsLoss()
        
        model=lightly.models.BarlowTwins(backbone,num_ftrs=num_ftrs)
        
        super().__init__(model,
                         criterion,
                         optim,
                         lr,
                         epoch,
                         num_ftrs,
                         dataloader_kNN,
                         num_classes
                         )
class LitBYOLSelf(LitSelfModel):
    def __init__(self,
                backbone:nn.modules,
                 optim: str,
                 lr: float,
                 epoch: int,
                 num_ftrs: int,
                 dataloader_kNN: torch.utils.data.DataLoader,
                 num_classes: int
                 ):
                
        criterion=lightly.loss.SymNegCosineSimilarityLoss()
        
        model=lightly.models.BYOL(backbone,num_ftrs=num_ftrs)
        
        super().__init__(model,
                         criterion,
                         optim,
                         lr,
                         epoch,
                         num_ftrs,
                         dataloader_kNN,
                         num_classes
                         )
class LitSimSiamSelf(LitSelfModel):
    def __init__(self,
                backbone:nn.modules,
                 optim: str,
                 lr: float,
                 epoch: int,
                 num_ftrs: int,
                 dataloader_kNN: torch.utils.data.DataLoader,
                 num_classes: int
                 ):
                
        criterion=lightly.loss.SymNegCosineSimilarityLoss()
        
        model=lightly.models.SimSiam(backbone,num_ftrs=num_ftrs,num_mlp_layers=2)
        
        super().__init__(model,
                         criterion,
                         optim,
                         lr,
                         epoch,
                         num_ftrs,
                         dataloader_kNN,
                         num_classes
                         )
class LitSimCLRSelf(LitSelfModel):
    def __init__(self,
                backbone:nn.modules,
                 optim: str,
                 lr: float,
                 epoch: int,
                 num_ftrs: int,
                 dataloader_kNN: torch.utils.data.DataLoader,
                 num_classes: int
                 ):
                
        criterion=lightly.loss.NTXentLoss()
        
        model=lightly.models.SimCLR(backbone,num_ftrs=num_ftrs)
        
        super().__init__(model,
                         criterion,
                         optim,
                         lr,
                         epoch,
                         num_ftrs,
                         dataloader_kNN,
                         num_classes
                         )
class LitMocoSelf(LitSelfModel):
    def __init__(self,
                    backbone:nn.modules,
                    optim: str,
                    lr: float,
                    epoch: int,
                    num_ftrs: int,
                    dataloader_kNN: torch.utils.data.DataLoader,
                    num_classes: int
                    ):
            
        memory_bank_size = 4096
        
        # create our loss with the optional memory bank
        criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)
        
        model=lightly.models.MoCo(
            backbone,
            num_ftrs=num_ftrs,
            m=0.99,
            # out_dim=out_dim,
        )
        
        super().__init__(model,
                            criterion,
                            optim,
                            lr,
                            epoch,
                            num_ftrs,
                            dataloader_kNN,
                            num_classes
                            )
    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*self.model(x0, x1))
        loss_2 = self.criterion(*self.model(x1, x0))
        loss = 0.5 * (loss_1 + loss_2)
        self.log('train_loss_ssl', loss)
        return loss

class LitClassifier(LitSystemTemplate):
    def __init__(self, model:LitSystemTemplate,optim:str,num_class:int=196):
        super().__init__(
            optim_name=optim,
            epoch=model.epochs,
            lr=model.lr,
        )
        # create a moco based on ResNet
        self.model = model.model

        # freeze the layers of moco
        for p in self.model.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(model.num_ftrs, num_class)  #hacer el 512 variable pero más adelante
        
    def forward(self, x):
        with torch.no_grad():
            y_hat = self.model.backbone(x)#.squeeze()  comprobar cual es la salida sin squeeze
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