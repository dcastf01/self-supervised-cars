

from dataclasses import asdict
import datetime
import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from builders import (get_callbacks, get_datamodule, get_self_system,
                      get_trainer, get_transform_collate_function,get_classifier_system)
from config import CONFIG


def main():
    os.environ["WANDB_IGNORE_GLOBS"]="*.ckpt"
    logging.info("empezando setup del experimento")
    torch.backends.cudnn.benchmark = True
    config=CONFIG()
    config_dict=asdict(config)
    wandb.init(
            project="Cars Semi-Supervised Image Classification",
            entity='dcastf01',
                            
            config=config_dict
                )
    config=wandb.config
    print(config)
    wandb.run.name=config.experiment_name[:5]+" "+\
                    datetime.datetime.utcnow().strftime("%b %d %X")
                    
    wandb.run.notes=config.notes
    # wandb.run.save()
    
    wandb_logger=WandbLogger(
        #offline=True,
        log_model =False
                )
    #get transform_fn
    
    transfrom_fn,transform_fn_test,collate_fn=get_transform_collate_function(
        config.experiment_name,
        config.IMG_SIZE,
        backbone_name=config.backbone_name
        
        )
    #get datamodule
    dm=get_datamodule(config.dataset_name,
                      config.data_dir,
                      config.batch_size,
                      transfrom_fn,
                      transform_fn_test,
                      collate_fn
                      )
    #get callbacks 
    callbacks=get_callbacks(config,dm)
    #get system
    self_system=get_self_system(   
                    experiment_name=config.experiment_name,
                    backbone_name=config.backbone_name,
                    optim=config.optim_name,
                    lr= config.lr,
                    img_size=config.IMG_SIZE,
                    epochs=config.NUM_EPOCHS_SELF_TRAIN,
                    dataloader_kNN=dm.trainclassifier_dataloader(),
                    num_classes=196
                        
                     )
    #create trainer
    self_trainer=get_trainer(wandb_logger,callbacks,
                             config.NUM_EPOCHS_SELF_TRAIN,
                             config)
    

    logging.info("empezando el entrenamiento")
    self_trainer.fit(self_system,datamodule=dm)
    
    classifier_system=get_classifier_system(self_system,
                                            config.optim_name,
                                            config.NUM_EPOCHS_CLASSIFIER_TRAIN,)
    
    classifier_trainer=get_trainer(wandb_logger,callbacks,
                                   config.NUM_EPOCHS_CLASSIFIER_TRAIN,
                                   config)
    classifier_trainer.fit(
            classifier_system,
            dm.trainclassifier_dataloader(),
            dm.test_dataloader()
                )
    # classifier_trainer.test(test_dataloaders=dm.test_dataloader())


if __name__=="__main__":
    main()
