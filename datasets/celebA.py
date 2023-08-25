"""
This file declare pytorch dataset to load CelebA image and attribute save in csv file
"""
import os
import csv
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_path,  transform=None):
        self.root_dir = root_dir
        self.attr_path = attr_path
        self.file_list = os.listdir(root_dir)
        self.transform = transform
        
        self.read_attr_file()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_infor = self.data[idx]
        image_name = data_infor[0]
        img_path = os.path.join(self.root_dir, image_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        attribute = [float(item) for item in data_infor[1:]]
        attribute = torch.tensor(attribute)

        sample = {
            'image': image,
            'filename': image_name,
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
    import matplotlib.pyplot as plt

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_dir = 'C:/MyLibrary/Data/img_align_celeba'
    attr_path = "C:/MyLibrary/Data/celebA_Anno/list_attr_celeba.csv"

    dataset = CelebADataset(root_dir=image_dir, attr_path=attr_path, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    
    for i, batch in enumerate(dataloader):
        images = batch['image']
        filenames = batch['filename']
        print(images[0].shape)
        # Transpose the tensor from (channels, height, width) to (height, width, channels)
        image_np = images[0].permute(1, 2, 0).numpy()

        # Display the image
        plt.imshow(image_np)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()
        print(batch['attribute'])
        # Your processing code here using images and filenames