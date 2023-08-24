
"""
This is a class with function use for training an monitoring model
class Attr_Trainer():
    def __init__(self, config) -> None:
        pass
        
    def repare_logger(self):
        pass
        
    def repare_data(self):
        pass

    def configure_optimization(self):
        pass

    def load_checkpoint(self):
        pass

    def save_model(self, save_best=False):
        pass

    def val_step(self):
        pass

    def compute_loss(self):
        pass

    def fit(self):
        pass

"""
import os
import torch
from loguru import logger
from models.my_CNN_model import My_CNN_models
from datasets.celebA import CelebADataset

class Attr_Trainer():
    def __init__(self, config):
        self.cfg = config

        self.model = My_CNN_models(output_dims=self.cfg.output_dims)

        self.repare_data()
        self.repare_logger()
        self.configure_optimization()

    def repare_logger(self):
        os.makedirs(self.cfg.output.dir, exist_ok=True)
        os.makedirs(self.cfg.output.log_dir, exist_ok=True)

        logger.add(os.path.join(self.cfg.log_dir, 'train.log'))
        pass

    def repare_data(self):
        self.train_dataset = CelebADataset()
        self.val_dataset = CelebADataset()
        logger.info(f'---- training data numbers: {len(self.train_dataset)}')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.finetex.num_worker,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
        self.val_iter = iter(self.val_dataloader)

    def configure_optimization(self):
        self.optim = torch.optim.AdamW(
                                self.model.parameters(),
                                lr=self.cfg.finetex.lr,)

    def load_checkpoint(self):
        pass

    def save_model(self, save_best=False):
        pass

    def val_step(self):
        pass

    def compute_loss(self):
        pass

    def fit(self):
        pass