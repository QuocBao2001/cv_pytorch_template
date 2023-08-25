
"""
This is a class with function use for training an monitoring model
class Attr_Trainer():
    def __init__(self, config) -> None:
        pass
        
    def prepare_logger(self):
        pass
        
    def prepare_data(self):
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
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from torch.utils.data import DataLoader

from models.my_CNN_model import My_CNN_models
from datasets.celebA import CelebADataset

class Attr_Trainer():
    def __init__(self, config):
        self.cfg = config

        # declare device, model and loss instance
        self.device = self.cfg.device + ':' + self.cfg.device_id
        self.model = My_CNN_models(output_dims=self.cfg.output_dims)
        self.BCE = torch.nn.BCELoss()

        self.prepare_logger()
        self.prepare_data()
        self.load_checkpoint()
        self.configure_optimization()

    def prepare_logger(self):
        """
        This function use to declare any logger using to monitor training
        """
        os.makedirs(self.cfg.output.dir, exist_ok=True)
        os.makedirs(self.cfg.output.log_dir, exist_ok=True)

        logger.add(os.path.join(self.cfg.log_dir, 'train.log'))
        pass

    def prepare_data(self):
        """
        This function use to declare dataset for training and validation
        """
        image_dir = self.cfg.data.image_dir
        self.train_dataset = CelebADataset(image_dir, self.cfg.data.train_csv_path)
        self.val_dataset = CelebADataset(image_dir, self.cfg.data.val_csv_path )

        logger.info(f'---- training data numbers: {len(self.train_dataset)}')
        logger.info(f'---- training data numbers: {len(self.train_dataset)}')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.Train.batch_size, shuffle=True,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.cfg.Train.batch_size, shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False)
        
        self.train_iter = iter(self.train_dataloader)
        self.val_iter = iter(self.val_dataloader)

    def configure_optimization(self):
        """
        This function use to declare optimization
        """
        self.optim = torch.optim.AdamW(
                                self.model.parameters(),
                                lr=self.cfg.finetex.lr,)

    def load_checkpoint(self):
        """
        This function use to load checkpoint from previous training
        """
        # total training loops have done
        self.global_step = 0

    def save_model(self, save_best=False):
        """
        This function use to save model weight and other
        """
        pass

    def val_step(self):
        """
        This function use in validation step
        """
        pass

    def compute_loss(self, infer_value, target_value):
        """
        This function use to compute loss function, seperate for helpful when deal with multi-sub loss
        """
        loss_value = self.BCE(infer_value, target_value)
        return loss_value

    def fit(self):
        """
        This is the main training function
        """
        # get total loops in one epochs
        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)

        # get the start epoch for continue training if have previous training session
        start_epoch = self.global_step//iters_every_epoch

        # for each epoch
        for epoch in range(start_epoch, self.cfg.Train.epochs):
            # for a step in a single epoch
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.Train.epochs}]"):
                # move to the step that fit with previous
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                
                ######################################## Training ########################################
                ### Step 1: get data dictionary ----------------------------------------------------------
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                
                images = batch['image'].to(self.device)
                target_code = batch['attribute'].to(self.device)

                ### Step 2: set model to training mode and infer -----------------------------------------
                self.model.train()
                infer_code = self.model.model(images)
                
                ### Step 3: call compute loss function ---------------------------------------------------
                loss = self.L2_loss(infer_code, target_code)
                self.opt.zero_grad()

                ### Step 4: backward and optims ----------------------------------------------------------
                loss.backward()
                self.opt.step()
                ##########################################################################################

                ### log result  --------------------------------------------------------------------------
                if self.global_step % self.cfg.Csup.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \
                                    \nEpoch: {epoch}, \
                                    Iter: {step}/{iters_every_epoch}, \
                                    Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
                    # for k, v in losses.items():
                    #     loss_info = loss_info + f'{k}: {v:.4f}, '
                   
                    loss_info = loss_info + f', total loss: {loss:.4f}, '
                    logger.info(loss_info)

                ### visualize result  --------------------------------------------------------------------
                if self.global_step % self.cfg.Csup.vis_steps == 0:
                    pass

                ### save checkpoint  ---------------------------------------------------------------------
                if self.global_step % self.cfg.Train.log_steps  == 0:
                    self.save_model()

                ### validate model   ---------------------------------------------------------------------
                if self.global_step % self.cfg.Train.val_steps == 0:
                    self.validation_step()
                
                # if self.global_step % self.cfg.Train.eval_steps == 0:
                #     self.evaluate()

                self.global_step += 1
                if self.global_step > self.cfg.Train.num_steps:
                    break