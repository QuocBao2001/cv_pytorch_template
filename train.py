import yaml
import shutil
import os, sys
import numpy as np

from attr_trainer import Attr_Trainer
from configs.default_configs import parse_args

def main(cfg):
    # creat folders 
    os.makedirs(cfg.output.dir, exist_ok=True)
    os.makedirs(cfg.output.log_dir, exist_ok=True)
    os.makedirs(cfg.output.ckpt_dir, exist_ok=True)

    with open(os.path.join(cfg.output.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    if cfg.cfg_file is not None:
        shutil.copy(cfg.cfg_file, os.path.join(cfg.output.log_dir, 'config.yaml'))
    

    # start training
    Trainer = Attr_Trainer(cfg)
    Trainer.fit()

if __name__ == '__main__':
    cfg = parse_args()
    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.exp_name = exp_name
    main(cfg)

# usage:
# python train.py --cfg configs/file/path.yml 