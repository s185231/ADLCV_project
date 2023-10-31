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
        self.path_img = os.path.join(_DATA_PATH, state, ev)
        self.path_target = os.path.join(_DATA_PATH, state, '0')
        self.transform = transform
        # import all images in the self.path directory
        self.files_img = os.listdir(self.path_img)
        self.files_target = os.listdir(self.path_target)
        self.images = []
        self.targets = []
        pbar_img = tqdm(self.files_img)
        pbar_target = tqdm(self.files_target)

        for file in pbar_img:
            img = self.transform(Image.open(os.path.join(self.path_img,file)))
            print(img.min(), img.max())
            self.ev_list.append(img)
        for file in pbar_target:
            target = self.transform(Image.open(os.path.join(self.path_target,file)))
            print(img.min(), img.max())
            self.target_list.append(target)
            self.target_list.append(target)
        print(len(self.ev_list), len(self.target_list))

    def __len__(self):
        return len(self.ev_list)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        print(img.min(), img.max())
        #img = self.transform(img)
        target = self.targets[idx]
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
