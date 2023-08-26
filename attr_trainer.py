
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
from torchvision import transforms
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader

from models.my_CNN_model import My_CNN_models
from datasets.celebA import CelebADataset

class Attr_Trainer():
    def __init__(self, config):
        self.cfg = config

        # declare device, model and loss instance
        self.device = self.cfg.device + ':' + self.cfg.device_id
        self.model = My_CNN_models(output_dims=self.cfg.model.output_dims).to(self.device)
        self.BCE = torch.nn.BCELoss()

        self.prepare_logger()
        self.prepare_data()
        self.configure_optimization()
        self.load_checkpoint()

        self.best_val_loss = 1e9

    def prepare_logger(self):
        """
        This function use to declare any logger using to monitor training
        """
        logger.add(os.path.join(self.cfg.output.log_dir, 'train.log'))
        if self.cfg.log.tensorboard:
            self.tb_train_logger = tb.SummaryWriter(os.path.join(self.cfg.output.log_dir, 'train'), flush_secs=1)
            self.tb_valid_logger = tb.SummaryWriter(os.path.join(self.cfg.output.log_dir, 'valid'), flush_secs=1)


    def prepare_data(self):
        """
        This function use to declare dataset for training and validation
        """
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image_dir = self.cfg.data.image_dir
        self.train_dataset = CelebADataset(image_dir, self.cfg.data.train_csv_path, data_transform)
        self.val_dataset = CelebADataset(image_dir, self.cfg.data.val_csv_path, data_transform)

        logger.info(f'---- training data numbers: {len(self.train_dataset)}')
        logger.info(f'---- validation data numbers: {len(self.val_dataset)}')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.Train.batch_size, shuffle=True,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.cfg.Train.batch_size, shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
        
        self.train_iter = iter(self.train_dataloader)
        self.val_iter = iter(self.val_dataloader)

    def configure_optimization(self):
        """
        This function use to declare optimization
        """
        self.optim = torch.optim.AdamW(
                                self.model.parameters(),
                                lr=self.cfg.Train.lr)

    def load_checkpoint(self, is_best=False):
        """
        This function use to load checkpoint from previous training
        """
        # total training loops have done
        self.train_global_step = 0
        self.val_global_step = 0
        # resume training, including model weight, opt, steps
        if self.cfg.Train.resume:
            if os.path.exists(self.cfg.Train.checkpoint_path):
                checkpoint = torch.load(self.cfg.Train.checkpoint_path)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optim.load_state_dict(checkpoint['opt'])
                self.train_global_step = checkpoint['train_global_step']
                self.val_global_step = checkpoint['val_global_step']
                # write log
                logger.info(f"resume training from {self.cfg.Train.checkpoint_path}")
                logger.info(f"training start from step {self.train_global_step}")
            else: 
                raise ValueError('checkpoint model path not found')
        else:
            if os.path.exists(self.cfg.Train.pretrained_path):
                logger.info('Train from pretrained')
                pretrained_weight = torch.load(self.cfg.Train.pretrained_path)
                self.model.load_state_dict(pretrained_weight)
                self.train_global_step = 0
            else:
                logger.info('can not find pretrained weight, training from scratch')
                self.train_global_step = 0

    def save_model(self, save_best=False):
        """
        This function use to save model weight and other
        """
        model_dict = {}
        model_dict['state_dict'] = self.model.state_dict()
        model_dict['opt'] = self.optim.state_dict()
        model_dict['train_global_step'] = self.train_global_step
        model_dict['val_global_step'] = self.val_global_step
        model_dict['batch_size'] = self.cfg.Train.batch_size
        if save_best:
            torch.save(model_dict, os.path.join(self.cfg.output.ckpt_dir, 
                                                f'best_at_{self.train_global_step:08}.tar'))  
        else:
            torch.save(model_dict, os.path.join(self.cfg.output.ckpt_dir, 
                                                f'{self.train_global_step:08}.tar'))   

    def val_step(self):
        """
        This function use in validation step
        """
        # change to eval mode
        self.model.eval()
        total_loss = 0
        num_iters = int(len(self.val_dataset)/self.cfg.Train.batch_size)
        # testing
        num_iters = 1000
        #for step in tqdm(range(num_iters), desc=f"Val iter"):
        for step in tqdm(range(1000), desc=f"Val iter"):
            try:
                batch = next(self.val_iter)
            except:
                self.val_iter = iter(self.val_dataloader)
                batch = next(self.val_iter)

            images = batch['image'].to(self.device)
            target_code = batch['attribute'].to(self.device)
            with torch.no_grad():
                infer_code = self.model(images)
                loss = self.compute_loss(infer_code, target_code)
                total_loss += loss
            self.val_global_step += 1
            if self.cfg.log.tensorboard:
                self.tb_valid_logger.add_scalar('valid/loss', loss, global_step=self.val_global_step)
        
        total_loss = total_loss / num_iters
        # log val total loss
        logger.info(f'validate loss: {total_loss:.4f}')

        # self best weight if total loss less than current best val loss
        if total_loss < self.best_val_loss:
            self.save_model(save_best=True)   
            self.best_val_loss = total_loss
        
        # set back model to train mode
        self.model.train()
        return loss

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
        iters_every_epoch = int(len(self.train_dataset)/self.cfg.Train.batch_size)

        # get the start epoch for continue training if have previous training session
        start_epoch = self.train_global_step//iters_every_epoch

        # set model to train mode
        self.model.train()

        # for each epoch
        for epoch in range(start_epoch, self.cfg.Train.epochs):
            # for a step in a single epoch
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.Train.epochs}]"):
                # move to the step that fit with previous
                if epoch*iters_every_epoch + step < self.train_global_step:
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

                ### Step 2: infer ------------------------------------------------------------------------
                infer_code = self.model(images)
                
                ### Step 3: call compute loss function ---------------------------------------------------
                loss = self.compute_loss(infer_code, target_code)
                self.optim.zero_grad()

                ### Step 4: backward and optims ----------------------------------------------------------
                loss.backward()
                self.optim.step()
                ##########################################################################################

                ### log result  --------------------------------------------------------------------------
                if self.train_global_step % self.cfg.Train.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \
                                    \nEpoch: {epoch}, \
                                    Iter: {step}/{iters_every_epoch}, \
                                    Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
                    # for k, v in losses.items():
                    #     loss_info = loss_info + f'{k}: {v:.4f}, '
                   
                    loss_info = loss_info + f', total loss: {loss:.4f}, '
                    logger.info(loss_info)
                if self.cfg.log.tensorboard:
                    self.tb_train_logger.add_scalar('train/loss', 
                                                                loss, global_step=self.train_global_step)

                ### visualize result  --------------------------------------------------------------------
                if self.train_global_step % self.cfg.Train.vis_steps == 0:
                    pass

                ### save checkpoint  ---------------------------------------------------------------------
                if self.train_global_step % self.cfg.Train.save_ckpt_steps  == 0:
                    self.save_model()

                ### validate model   ---------------------------------------------------------------------
                if self.train_global_step % self.cfg.Train.val_steps == 0:
                    self.val_step()
                
                # if self.global_step % self.cfg.Train.eval_steps == 0:
                #     self.evaluate()

                self.train_global_step += 1
                if self.train_global_step > self.cfg.Train.num_steps:
                    break