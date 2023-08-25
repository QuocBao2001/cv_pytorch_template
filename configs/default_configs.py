'''
Default config
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os
import torch

cfg = CN()

# data config
cfg.data = CN()
cfg.data.label_name = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs",
                    "Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows",
                    "Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones",
                    "Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin",
                    "Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
                    "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace",
                    "Wearing_Necktie","Young"]

cfg.data.image_dir = 'C:/MyLibrary/Data/img_align_celeba'
cfg.data.train_csv_path = "C:/MyLibrary/Data/celebA_Anno/list_attr/output_1.csv"
cfg.data.val_csv_path = "C:/MyLibrary/Data/celebA_Anno/list_attr/output_2.csv"
cfg.data.test_csv_path = "C:/MyLibrary/Data/celebA_Anno/list_attr/output_3.csv"

# output config
cfg.output = CN()
cfg.output.dir = "C:/MyLibrary/Pytorch_Template/first_run"
cfg.output.log_dir = os.path.join(cfg.output_dir, "logs")
cfg.output.ckpt_dir = os.path.join(cfg.output_dir, "ckpt")

# model config
cfg.Model == CN()
cfg.Model.output_dims = len(cfg.data.label_name)

# training config
cfg.Train = CN()
cfg.Train.device = 'cuda'
cfg.Train.device_id = '0'
cfg.Train.batch_size = 8
cfg.Train.img_size = 224
cfg.Train.epochs = 50
cfg.Train.log_steps = 100
cfg.Train.val_log_steps = 10
cfg.Train.save_ckpt_steps = 500
cfg.Train.val_steps = 500
cfg.Train.vis_steps = 1000
cfg.Train.num_steps = 100000000

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    """Update config by external file via parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
