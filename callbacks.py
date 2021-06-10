



from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as functional
#
from matplotlib import rcParams as rcp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb
 
class PlotLatentSpace(Callback):
    def __init__(self,dataloader:DataLoader) -> None:
        super(PlotLatentSpace,self).__init__()
        self.dataloader=dataloader
        self.path_to_data=dataloader.dataset.root_images
        self.num_labels=5
    def upload_image(self,trainer,image,text:str):
       
        trainer.logger.experiment.log({
            f"{text}/latent_space": [
                wandb.Image(image) 
                ],
            })
    def create_embbedings_2d(self,trainer,pl_module):
        embeddings = []
        filenames = []
        labels=[]
        rng=np.random.default_rng()
        class_selected=torch.from_numpy(rng.choice(196,size=self.num_labels,replace=False))
        # disable gradients for faster calculations
        pl_module.eval()
        with torch.no_grad():
            for i, (x, y_true, fnames) in enumerate(self.dataloader):
                # move the images to the gpu
                x = x.to(device=pl_module.device)
                # embed the images with the pre-trained backbone
                y = pl_module.model.backbone(x)
                # y = y.squeeze()
                for embbeding,label in zip(y,y_true):
                    if class_selected  in label :
                        # store the embeddings and filenames in lists
                        embeddings.append(torch.unsqueeze(embbeding,dim=0))
                        # filenames = filenames + list(fnames)
                        labels.append(label.item())
                # if i*x.shape[0]>250:
                #     break

        # concatenate the embeddings and convert to numpy
        embeddings = torch.cat(embeddings, dim=0)
        
        embeddings = embeddings.cpu()
        tl=TSNE()
        embeddings_2d=tl.fit_transform(embeddings)
        # projection = random_projection.GaussianRandomProjection(n_components=2)
        # embeddings_2d = projection.fit_transform   (embeddings)
        return embeddings_2d ,labels
    
    def plot_emmbedings(self,trainer,pl_module,embedding,labels):
        fig=plt.figure(figsize=(10,10))

        color_pallete=sns.color_palette("tab10")[:self.num_labels]
        sns.scatterplot(embedding[:,0], embedding[:,1], hue=labels,palette=color_pallete)
        fig.savefig('ax2_figureonlyxlabel.png')
        if hasattr( pl_module,"fc"):
            text="classifier"
        else:
            text="Self"
        self.upload_image(trainer,fig,text)
        pass
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        embeddings_2d,filenames=self.create_embbedings_2d(trainer,pl_module)
        self.plot_emmbedings(trainer,pl_module,embeddings_2d,filenames)
   
        return super().on_train_end(trainer, pl_module) 

class AccuracyEnd(Callback):
    
    def __init__(self,dataloader:DataLoader,prefix=None) -> None:
        super(AccuracyEnd,self).__init__()
        self.dataloader=dataloader
        
    def generate_accuracy_and_upload(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        all_results=[]
        all_targets=[]
        num_correct = 0
        num_samples = 0
        for batch in self.dataloader:
            image,target,idx=batch
            y=target[0].to(pl_module.device)
            with torch.no_grad():
                # results=pl_module(image.to(device=pl_module.device))
                scores = pl_module(image.to(device=pl_module.device))
                _, predictions = scores.max(1)
                num_correct +=torch.sum(predictions == y)
                num_samples += predictions.size(0)
            # all_results.append(results.softmax(dim=1))
            # all_targets.append(target)
        accuracy=num_correct/num_samples
        # accuracy = self.simple_accuracy(all_results, all_targets)
        trainer.logger.experiment.log({
            "Accuracy ":accuracy.item(),
        
                })

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.generate_accuracy_and_upload(trainer,pl_module)
   
        return super().on_train_end(trainer, pl_module)
    
        
    def simple_accuracy(self,preds, labels):
        return (preds == labels).mean()
    
    
