
import logging

import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from callbacks import AccuracyEnd
from config import (CONFIG,   Dataset,
                 ModelsAvailable,ExperimentAvailable
                    )
from data_modules import (Cars196DataModule, FGVCAircraftDataModule,
                          GroceryStoreDataModule)
from factory_augmentations import (get_test_transforms,get_normalize_parameter_by_model,get_train_classifier_transforms)
from factory_collate import (collate_mixup, collate_triplet_loss,
                             collate_two_images)
from lit_models import LitSelfModel,LitClassifier
from factory_backbone import create_backbone
from losses import SymNegCosineSimilarityLoss,BarlowTwinsLoss,NTXentLoss
import lightly

def setup_experiment(experiment_name:str,model_name:str,img_size,optim_name,lr:float,steps_per_epoch=100):#arreglar steps per epoch
    
    pass
def get_transform_collate_function(experiment_name:str,
                                   img_size:int,
                                   backbone_name:str,
                           ):
    experiment_enum=ExperimentAvailable[experiment_name.lower()]
    
    backbone_enum=ModelsAvailable[backbone_name.lower()]
    
    mean,std,interpolation=get_normalize_parameter_by_model(backbone_enum)

    collate_fn=None
    if experiment_enum==ExperimentAvailable.simsiam:
        # define the augmentations for self-supervised learning
        collate_fn = lightly.data.ImageCollateFunction(
            input_size=img_size,
            # require invariance to flips and rotations
            hf_prob=0.5,
            vf_prob=0.5,
            rr_prob=0.5,
            # satellite images are all taken from the same height
            # so we use only slight random cropping
            min_scale=0.5,
            # use a weak color jitter for invariance w.r.t small color changes
            cj_prob=0.2,
            cj_bright=0.1,
            cj_contrast=0.1,
            cj_hue=0.1,
            cj_sat=0.1,
        )
       
    elif experiment_enum==ExperimentAvailable.simclr:
        pass
    elif experiment_enum==ExperimentAvailable.moco:

        collate_fn = lightly.data.SimCLRCollateFunction(
            input_size=448,
                            )

    elif experiment_enum==ExperimentAvailable.barlowtwins:
        pass
    elif experiment_enum==ExperimentAvailable.byol:
        pass
    
    elif experiment_enum==ExperimentAvailable.nnclr:
        pass
    
    
    transform_fn_classification=get_train_classifier_transforms(img_size,
                                            mean,
                                            std,
                                            interpolation)
    
    # No additional augmentations for the test set
    transform_fn_test=get_test_transforms(img_size,
                                            mean,
                                            std,
                                            interpolation)
    return transform_fn_classification,transform_fn_test,collate_fn
    
def get_datamodule(name_dataset:str
                   ,batch_size:int,
                   transform_fn,
                   transform_fn_test,
                   collate_fn
                   ):

    if isinstance(name_dataset,str):
        name_dataset=Dataset[name_dataset.lower()]
    
    if name_dataset==Dataset.grocerydataset:
        dm=GroceryStoreDataModule(
            data_dir="data",
            batch_size=batch_size,
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            collate_fn=collate_fn
            )
        
    elif name_dataset==Dataset.fgvcaircraft:
        dm=FGVCAircraftDataModule(
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            data_dir="data",
            batch_size=batch_size,
            collate_fn=collate_fn
            )
    
    elif name_dataset==Dataset.cars196:
        dm=Cars196DataModule(
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            data_dir="data",
            batch_size=batch_size,
            collate_fn=collate_fn
                    )
    
    else: 
        raise ("choice a correct dataset")
    
    dm.prepare_data()   
    dm.setup()
    return dm

# def get_losses_fn( config)->dict:
#     losses_fn={}
    
#     if config.loss_crossentropy_standar:
#         losses_fn["crossentropy"]=CrosentropyStandar()
#     if config.loss_contrastive_standar:
#         losses_fn["contrastive_standar"]=NotImplementedError
#     if config.loss_contrastive_fg:
#         losses_fn["contrastive_fg"]=ContrastiveLossFG()
#     if config.loss_cosine_similarity_simsiam:
#         losses_fn["similarity"]=SymNegCosineSimilarityLoss()
#     if config.loss_triplet:
#         losses_fn["triplet_loss"]=TripletMarginLoss()
        
#     if len(losses_fn)==0:
#         raise("select unless one loss")
    
#     return losses_fn

def get_callbacks(config,dm):
    #callbacks
    
    early_stopping=EarlyStopping(monitor='_valid_level0Accuracy',
                                 mode="max",
                                patience=10,
                                 verbose=True,
                                 check_finite =True
                                 )

    checkpoint_callback = ModelCheckpoint(
        monitor='_val_loss',
        dirpath=config.PATH_CHECKPOINT,
        filename= '-{epoch:02d}-{val_loss:.6f}',
        mode="min",
        save_last=True,
        save_top_k=3,
                        )
    learning_rate_monitor=LearningRateMonitor(logging_interval="epoch")
    
    accuracytest=AccuracyEnd(dm.test_dataloader(),prefix="test")
    # freeze_layers_name=config.freeze_layers_name
    # freeze_layer_enum=FreezeLayersAvailable[freeze_layers_name.lower()]
    # if freeze_layer_enum==FreezeLayersAvailable.none:
    #     callbacks=[
    #     accuracytest,
    #     learning_rate_monitor,
    #     early_stopping,
    #         ]
    # else:
    #     freeze_layers=FreezeLayers(freeze_layer_enum)
    callbacks=[
        # accuracytest,
        learning_rate_monitor,
        # early_stopping,
        # freeze_layers
            ]
        
    return callbacks

def get_self_system( experiment_name:str,
                backbone_name:str,
                optim:str,
                lr:float,
                img_size:int,
                epochs:int,
                steps_per_epoch:int,
                

               ):
    
    experiment_enum=ExperimentAvailable[experiment_name.lower()]
    backbone_enum=ModelsAvailable[backbone_name.lower()]
    
    backbone,num_ftrs=create_backbone(backbone_enum,img_size=img_size)
    if experiment_enum==ExperimentAvailable.simsiam:
        # dimension of the embeddings
        
        # dimension of the output of the prediction and projection heads
        out_dim = proj_hidden_dim = 2048
        # the prediction head uses a bottleneck architecture
        pred_hidden_dim = 512
        # use 2 layers in the projection head
        num_mlp_layers = 3
        # define the augmentations for self-supervised learning
       
        criterion=lightly.loss.SymNegCosineSimilarityLoss()
        model=lightly.models.SimSiam(
            backbone,
            num_ftrs=num_ftrs,
            proj_hidden_dim=proj_hidden_dim,
            pred_hidden_dim=pred_hidden_dim,
            out_dim=out_dim,
            num_mlp_layers=num_mlp_layers
        )

    elif experiment_enum==ExperimentAvailable.simclr:
       
        criterion=lightly.loss.NTXentLoss()
        
        model=lightly.models.SimCLR(backbone,num_ftrs=num_ftrs)
        
    elif experiment_enum==ExperimentAvailable.moco:
        out_dim  = 128
        memory_bank_size = 4096
        
        # create our loss with the optional memory bank
        criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)
        model=lightly.models.MoCo(
            backbone,
            num_ftrs=num_ftrs,
            m=0.99,
            out_dim=out_dim,
        )
     
    elif experiment_enum==ExperimentAvailable.barlowtwins:
        pass
    elif experiment_enum==ExperimentAvailable.byol:
        pass
    
    elif experiment_enum==ExperimentAvailable.nnclr:
        pass
    
    else:
        raise NotImplementedError
    
    self_system=LitSelfModel(
            model=model,
            criterion=criterion,
            optim=optim,
            lr=lr,
            epoch=epochs,
            steps_per_epoch=steps_per_epoch
                                )
    
    return self_system
def get_classifier_system(self_system:LitSelfModel,optim_model):
    
    return LitClassifier(self_system)
    
def get_trainer(wandb_logger,
                callbacks:list,
                config):
    
    gpus=[]
    if config.gpu0:
        gpus.append(0)
    if config.gpu1:
        gpus.append(1)
    logging.info( "gpus active",gpus)
    if len(gpus)>=2:
        distributed_backend="ddp"
        accelerator="dpp"
        plugins=DDPPlugin(find_unused_parameters=False)
    else:
        distributed_backend=None
        accelerator=None
        plugins=None
        
    trainer=pl.Trainer(
                    accumulate_grad_batches=5,
                    logger=wandb_logger,
                       gpus=gpus,
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.1, #only to debug
                    #    limit_val_batches=0.05, #only to debug
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,
                       log_gpu_memory=True,
                       distributed_backend=distributed_backend,
                       accelerator=accelerator,
                       plugins=plugins,
                       callbacks=callbacks,
                       progress_bar_refresh_rate=5,
                       
                       )
    
    return trainer

