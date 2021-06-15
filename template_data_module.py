import pytorch_lightning as pl
import os
from torchvision.datasets.folder import default_loader

class TemplateDataModule(pl.LightningDataModule):
    def __init__(self,
                 transform_fn,
                 transform_fn_test,
                 
                 classlevel:dict,
                 data_dir: str = "data",
                 batch_size: int = 32,
                 collate_fn=None
                 ):

        self.transform_fn=None
        self.transform_fn_classifier=transform_fn_test #poner cuando cuadre transform_fn ya que el cambio ha sido para una prueba
        self.transform_fn_test=transform_fn_test
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.classlevel=classlevel
        self.loader=default_loader
        self.collate_fn=collate_fn
        if collate_fn is not None:
            self.transform_fn=None
        super().__init__()
        
    # def prepare_data(self) -> None:
    #     return NotImplementedError    
    # def setup(self, stage=None):
    #     return NotImplementedError

    # def train_dataloader(self):
    #     return NotImplementedError

    # def val_dataloader(self):
    #     return NotImplementedError

    # def test_dataloader(self):
    #     return NotImplementedError