import torch.nn as nn
import torch.nn.functional as F

from config import ModelsAvailable
import timm
import numpy as np

def create_backbone(model_chosen:ModelsAvailable,
                img_size:int,
                pretrained:bool=True,
                    ):
            
            prefix_name=model_chosen.name[0:3]
            if prefix_name==ModelsAvailable.resnet50.name[0:3]:
                model=timm.create_model(model_chosen.value,pretrained=pretrained,num_classes=0,) #quizá habría que añadir global pool
                backbone=nn.Sequential(*list(model.children())[:-1])
                num_features=model.num_features
                
            elif prefix_name==ModelsAvailable.tf_efficientnet_b4_ns.name[0:3]:
                model=timm.create_model(model_chosen.value,pretrained=pretrained,num_classes=0, )#quizá habría que añadir global pool
                cosa=list(model.children())
                backbone=nn.Sequential(*list(model.children())[:-1])
                num_features=model.num_features
                
            else:
                NotImplementedError
                
            return backbone,num_features

