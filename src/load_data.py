import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split, Dataset
from PIL import Image
import os
from src import _DATA_PATH
from tqdm import tqdm
import random


class ExposureDataset(Dataset):
    def __init__(self, state, ev, transform, testing = False):
        if len(ev) == 2:
            self.path = os.path.join(_DATA_PATH, state, ev[0], ev[1])
            self.files_img = os.listdir(self.path)
            # make list of paths to images
            self.path_img = []
            for file in self.files_img:
                self.path_img.append(os.path.join(self.path,file))

        elif len(ev) == 1:
            self.path1 = os.path.join(_DATA_PATH, state, ev[0], '1')
            self.path2 = os.path.join(_DATA_PATH, state, ev[0], '2')

            self.files_img1 = os.listdir(self.path1)
            self.files_img2 = os.listdir(self.path2)

            self.path_img = []
            for i in range(len(self.files_img1)):
                r = random.randint(1,2)
                if r == 1:
                    file = self.files_img1[i]
                    self.path_img.append(os.path.join(self.path1,file))
                else:
                    file = self.files_img2[i]
                    self.path_img.append(os.path.join(self.path2,file))
        else:
            self.path1 = os.path.join(_DATA_PATH, state, 'P', '1')
            self.path2 = os.path.join(_DATA_PATH, state, 'P', '2')
            self.path3 = os.path.join(_DATA_PATH, state, 'N', '1')
            self.path4 = os.path.join(_DATA_PATH, state, 'N', '2')

            self.files_img1 = os.listdir(self.path1)
            self.files_img2 = os.listdir(self.path2)
            self.files_img3 = os.listdir(self.path3)
            self.files_img4 = os.listdir(self.path4)

            self.path_img = []
            for i in range(len(self.files_img1)):
                r = random.randint(1,4)
                if r == 1:
                    file = self.files_img1[i]
                    self.path_img.append(os.path.join(self.path1,file))
                elif r == 2:
                    file = self.files_img2[i]
                    self.path_img.append(os.path.join(self.path2,file))
                elif r == 3:
                    file = self.files_img3[i]
                    self.path_img.append(os.path.join(self.path3,file))
                else:
                    file = self.files_img4[i]
                    self.path_img.append(os.path.join(self.path4,file))

        self.path_target = os.path.join(_DATA_PATH, state, '0')
        self.transform = transform

        
        if testing:
            self.path_img = self.path_img[:10]
        
            
        self.images = []
        self.targets = []
        pbar_img = tqdm(self.path_img)

        for path in pbar_img:
            img = self.transform(Image.open(path))

            target_file = os.path.split(path)[-1]
            #print(target_file)
            target_file = target_file.split('_')[:-1]
            #print(target_file)
            target_file = '_'.join(target_file) + '_0.JPG'
            #print(target_file)

            target = self.transform(Image.open(os.path.join(self.path_target,target_file)))
            self.images.append(img)
            self.targets.append(target)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        return img, target

def get_dataloaders(ev, batch_size, image_size = 512, testing = False):
    assert (ev == 'P1' or ev == 'N1' or ev == 'P2' or ev == 'N2' or 'P' or 'N' or 'both'), "Please input P1, P2, N1 or N2"

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
    assert (ev == 'P1' or ev == 'N1' or ev == 'P2' or ev == 'N2' or 'P' or 'N' or 'both'), "Please input P1, P2, N1 or N2"

    data_transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((image_size, image_size), antialias=None),
        ]
    )
    testset = ExposureDataset("testing", ev, transform=data_transform_test, testing = testing)

    testloader = DataLoader(testset, batch_size=batch_size, num_workers=8, shuffle=False)

    return testloader


# Dataset from: https://github.com/Ryota-Kawamura/How-Diffusion-Models-Work
class SpritesDataset(Dataset):
    def __init__(self, ev, transform, testing = False):
        self.path_img = os.path.join(_DATA_PATH, 'sprites.npy')

        self.imgs = np.load(self.path_img)
        print(f"Dataset shape: {self.imgs.shape}")

        if testing:
            self.imgs = self.imgs[:10]
        
        self.transform = transform

        self.images = []
        self.targets = []

        pbar_img = tqdm(self.imgs)

        for img in pbar_img:
            img = img/255
            target = self.transform(img.astype(np.float32))
            
            if ev == 'P1':
                img = 2*img
                img = np.clip(img, 0, 1)
            elif ev == 'N1':
                img = img/2
            if ev == 'P2':
                gamma = 2
                img = img**(1/gamma)
            elif ev == 'N2':
                gamma = 0.5
                img = img**(1/gamma)
            elif ev == 'both':
                r = random.random()
                if r < 0.5:
                    img = 2*img
                    img = np.clip(img, 0, 1)
                else:
                    img = img/2
            else:
                gamma = random.random()*2
                img = img**(1/gamma)

            img = img.astype(np.float32)
            img = self.transform(img)
            self.images.append(img)
            self.targets.append(target)
        
        print(f"Dataset shape: {len(self.images)}")
                
    # Return the number of images in the dataset
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        return img, target



def get_Sprites_dataloaders(ev, batch_size, testing = False):
    assert (ev == 'P1' or ev == 'N1' or ev == 'P2' or ev == 'N2' or 'both' or 'both_gamma'), "Please input P1, P2, N1, N2 or both"

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            #transforms.RandomRotation(90),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            #transforms.GaussianBlur(kernel_size=5),
            #transforms.Resize((image_size, image_size), antialias=None),
        ]
    )


    dataset = SpritesDataset(ev, transform=data_transform, testing = testing)
    
    # Split dataset into train and validation and test with a fixed seed
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))


    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=8, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=8, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=8, shuffle=False)

    return trainloader, valloader, testloader

