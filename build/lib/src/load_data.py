import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split, Dataset
from PIL import Image
import os
from src import _DATA_PATH

class ExposureDataset(Dataset):
    def __init__(self,state,ev, transform):
        self.path = os.path.join(_DATA_PATH, state, "INPUT_IMAGES")
        self.transform = transform
        self.files = os.listdir(self.path)

        self.ev_list = []
        self.target_list = []
        for file in self.files:
            if file.split('_')[-1][0] == ev:
                self.ev_list.append(os.path.join(self.path,file))
            elif file.split('_')[-1][0] == '0':
                self.target_list.append(os.path.join(self.path,file))
                self.target_list.append(os.path.join(self.path,file))
        
    def __len__(self):
        return len(self.ev_list)
        
    def __getitem__(self, idx):
        img = Image.open(self.ev_list[idx])
        img = self.transform(img)
        target = Image.open(self.target_list[idx])
        target = self.transform(target)
        return img, target

def get_dataloaders(ev, batch_size, num_workers=8):
    assert (ev == 'P' or ev == 'N'), "Please input P or N"

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            #transforms.GaussianBlur(kernel_size=5),
            transforms.Resize((512, 512), antialias=None)
        ]
    )
    data_transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((512, 512), antialias=None)
        ]
    )

    trainset = ExposureDataset("training", ev, transform=data_transform)
    #print(len(trainset))
    valset = ExposureDataset("validation", ev, transform=data_transform_test)
    testset = ExposureDataset("testing", ev, transform=data_transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, valloader, testloader
