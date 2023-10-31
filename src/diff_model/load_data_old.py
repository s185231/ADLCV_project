import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split, Dataset
from PIL import Image
import os
from src import _DATA_PATH
from tqdm import tqdm

class ExposureDataset(Dataset):
    def __init__(self, state, ev, transform):
        self.path = os.path.join(_DATA_PATH, state, "INPUT_IMAGES")
        self.transform = transform
        self.files = os.listdir(self.path)
        self.ev_list = []
        self.target_list = []
        pbar = tqdm(self.files)

        for file in pbar:
            if file.split('_')[-1][0] == ev:
                img = self.transform(Image.open(os.path.join(self.path,file)))
                self.ev_list.append(img)
            elif file.split('_')[-1][0] == '0':
                target = self.transform(Image.open(os.path.join(self.path,file)))
                self.target_list.append(target)
                self.target_list.append(target)
        print(len(self.ev_list), len(self.target_list))
        
    def __len__(self):
        return len(self.ev_list)
        
    def __getitem__(self, idx):
        img = self.ev_list[idx]
        #img = self.transform(img)
        target = self.target_list[idx]
        #target = self.transform(target)
        return img, target

def get_dataloaders(ev, batch_size, image_size = 512):
    assert (ev == 'P' or ev == 'N'), "Please input P or N"

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            #transforms.RandomRotation(90),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            #transforms.GaussianBlur(kernel_size=5),
            transforms.Resize((image_size, image_size), antialias=None),
        ]
    )
    data_transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((image_size, image_size), antialias=None),
        ]
    )

    trainset = ExposureDataset("training", ev, transform=data_transform)
    valset = ExposureDataset("validation", ev, transform=data_transform_test)
    testset = ExposureDataset("testing", ev, transform=data_transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader
