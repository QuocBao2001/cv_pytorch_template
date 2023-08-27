import os
import torch
import argparse
from tqdm import tqdm
from loguru import logger
from torchvision import transforms
from torch.utils.data import DataLoader
from configs.default_configs import parse_args

from utils.utils import *
from models.my_CNN_model import My_CNN_models
from datasets.celebA import CelebADataset

def load_model(cfg):
    model = My_CNN_models(output_dims=cfg.model.output_dims).to(cfg.device)
    if os.path.exists(cfg.Train.checkpoint_path):
        logger.info('Infer by checkpoint')
        pretrained_weight = torch.load(cfg.Train.checkpoint_path)
        model.load_state_dict(pretrained_weight["state_dict"])
    else:
        raise ValueError('checkpoint model path not found')
    return model

def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_dir = cfg.data.image_dir
    test_dataset = CelebADataset(image_dir, cfg.data.test_csv_path, data_transform)

    logger.info(f'---- test data numbers: {len(test_dataset)}')

    test_dataloader = DataLoader(test_dataset, batch_size=cfg.Train.batch_size, shuffle=False,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=True)
    
    return test_dataloader

def main(cfg):
    # create folder
    os.makedirs(cfg.output.img_dir, exist_ok=True)
    # load model
    model = load_model(cfg)
    model.eval()
    # load data
    test_dataloader = load_data(cfg)

    # list save
    image_paths = []
    selected_labels_list = []
    list_size = 0
    img_idx = 0

    for batch in tqdm(test_dataloader, total=len(test_dataloader), desc=f"infer progress:"):
        images = batch['image'].to(cfg.device)
        img_names =  batch['filename']
        target_code = batch['attribute'].to(cfg.device)
        infer_vecs = model(images)
        for i, vec in enumerate(infer_vecs):
            vec_list = vec.cpu().detach().numpy().tolist()
            class_list = transform_vector_to_class(vector=vec_list, 
                                                   label_names=cfg.data.label_name, 
                                                   threshold=0.7)
            
            image_paths.append(os.path.join(cfg.data.image_dir, img_names[i]))
            selected_labels_list.append(class_list)
            list_size += 1

            if list_size == cfg.Infer.Imgs_per_grid:
                output_path = os.path.join(cfg.output.img_dir,  f'{img_idx}.jpg')
                add_labels_to_images(image_paths, selected_labels_list, output_path, cols=cfg.Infer.grid_cols)
                img_idx += 1
                image_paths = []
                selected_labels_list = []
                list_size = 0



if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    cfg = parse_args()
    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.exp_name = exp_name
    logger.add(os.path.join(cfg.output.log_dir, 'train.log'))
    main(cfg)
