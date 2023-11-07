import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split, Dataset
from PIL import Image
import os
from src import _DATA_PATH
from tqdm import tqdm

class ExposureDataset(Dataset):
    def __init__(self, state, ev, transform, testing = False):
        self.path_img = os.path.join(_DATA_PATH, state, ev[0], ev[1])
        self.path_target = os.path.join(_DATA_PATH, state, '0')
        self.transform = transform
        self.files_img = os.listdir(self.path_img)
        self.files_target = os.listdir(self.path_target)
        if testing:
            self.files_img = self.files_img[:10]
            self.files_target = self.files_target[:10]
        self.images = []
        self.targets = []
        pbar_img = tqdm(self.files_img)
        pbar_target = tqdm(self.files_target)

        for file in pbar_img:
            img = self.transform(Image.open(os.path.join(self.path_img,file)))
            self.images.append(img)
        for file in pbar_target:
            target = self.transform(Image.open(os.path.join(self.path_target,file)))
            self.targets.append(target)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        return img, target

def get_dataloaders(ev, batch_size, image_size = 512, testing = False):
    assert (ev == 'P1' or ev == 'N1' or ev == 'P2' or ev == 'N2'), "Please input P1, P2, N1 or N2"

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

    trainset = ExposureDataset("training", ev, transform=data_transform, testing = testing)
    valset = ExposureDataset("validation", ev, transform=data_transform_test, testing = testing)
    #testset = ExposureDataset("testing", ev, transform=data_transform_test, testing = testing)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=8, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=8, shuffle=False)
    #testloader = DataLoader(testset, batch_size=batch_size, num_workers=8, shuffle=False)

    return trainloader, valloader#, testloader



def get_test_dataloader(ev, batch_size, image_size = 512, testing = False):
    assert (ev == 'P1' or ev == 'N1' or ev == 'P2' or ev == 'N2'), "Please input P1, P2, N1 or N2"

    data_transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((image_size, image_size), antialias=None),
        ]
    )
    testset = ExposureDataset("testing", ev, transform=data_transform_test, testing = testing)

    testloader = DataLoader(testset, batch_size=batch_size, num_workers=8, shuffle=False)

    return testloader
