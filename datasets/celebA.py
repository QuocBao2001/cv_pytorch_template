import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, attr_path,  transform=None):
        self.root_dir = root_dir
        self.attr_path = attr_path
        self.file_list = os.listdir(root_dir)
        self.transform = transform
        
        self.read_attr_file()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_infor = self.data[idx]
        image_name = data_infor[0]
        img_path = os.path.join(self.root_dir, image_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        attribute = torch.tensor(data_infor[1:])

        sample = {
            'image': image,
            'attribute': attribute
        }
        
        return sample
    
    def read_attr_file(self):
        # read data infor from csv file
        self.data = []
        with open(self.attr_path, 'r') as file:
            csv_reader = csv.reader(file)
            # skip head row if need
            header = next(csv_reader)

            for row in csv_reader:
                self.data.append(row)



# Example usage
if __name__ == "__main__":
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageFolderDataset(root_dir='path/to/your/image/folder', transform=data_transform)
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    
    for i, batch in enumerate(dataloader):
        images = batch['image']
        filenames = batch['filename']
        
        # Your processing code here using images and filenames