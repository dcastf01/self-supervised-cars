import torch
import os
from enum import Enum

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""

class ModelsAvailable(Enum):
  
    resnet50="resnet50"
    resnet101="resnet101"
    tf_efficientnet_b4_ns="tf_efficientnet_b4_ns"
    vit_base_patch16_224_miil_in21k="vit_base_patch16_224_miil_in21k"

    
class Dataset (Enum):
    grocerydataset=1
    fgvcaircraft=2
    cars196=3
class Optim(Enum):
    adam=1
    sgd=2
class ExperimentAvailable(Enum):
    simsiam=1
    simclr=2
    moco=3
    barlowtwins=4
    byol=5
    nnclr=6
    

@dataclass(init=True)
class CONFIG(object):
    
    experiment=ExperimentAvailable.moco
    experiment_name:str=experiment.name
    
    backbone=ModelsAvailable.resnet50
    backbone_name:str=backbone.name

    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size:int =50
    
    dataset=Dataset.cars196
    dataset_name:str=dataset.name
    precision_compute:int=16
    
    optim=Optim.sgd
    optim_name:str=optim.name
    
    lr:float = 0.01
    AUTO_LR :bool= False
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 5
    SEED:int=1
    IMG_SIZE:int=448
    NUM_EPOCHS_SELF_TRAIN :int= 200 #poner 200
    NUM_EPOCHS_CLASSIFIER_TRAIN:int=20
    data_dir="/home/dcast/data/"
    # LOAD_MODEL :bool= True
    # SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"classification/model/checkpoint")
    
    gpu0:bool=True
    gpu1:bool=False
    notes:str="experimento moco probando lr de 0.1"
    ignore_globs:str="*.ckpt"
     
    #hyperparameters
    
