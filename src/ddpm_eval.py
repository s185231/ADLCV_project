import os
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

# custom imports
from ddpm import Diffusion
from model import UNet

import wandb
import argparse

from src.load_data import get_test_dataloader, get_Sprites_dataloaders


def save_images(images, originals, targets, path, show=True, title=None):
    images = 0.5*images + 0.5
    originals = 0.5*originals + 0.5
    targets = 0.5*targets + 0.5
    images = images.clamp(0, 1)
    originals = originals.clamp(0, 1)
    targets = targets.clamp(0, 1)
    fig, ax = plt.subplots(images.shape[0], 3, figsize=(3, images.shape[0]))
    ax[0, 0].set_title('input')
    ax[0, 1].set_title('output')
    ax[0, 2].set_title('target')
    for i in range(images.shape[0]):
        ax[i, 0].imshow(originals[i].permute(1, 2, 0).detach().cpu().numpy())
        ax[i, 0].axis('off')
        ax[i, 1].imshow(images[i].permute(1, 2, 0).detach().cpu().numpy())
        ax[i, 1].axis('off')
        ax[i, 2].imshow(targets[i].permute(1, 2, 0).detach().cpu().numpy())
        ax[i, 2].axis('off')
    
    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(wspace=0, hspace=0, top=0.85)
    if path is not None:
        fig.savefig(path, bbox_inches='tight', pad_inches=0)


class VGG(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        self.features = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features[:10]
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        #self.dropout = nn.Dropout(0.5)
        #self.fc = nn.Linear(256, num_classes)

    def forward(self, x, features=False):
        feat = self.features(x)
        feat = self.avg_pool(feat)
        #feat = self.flatten(feat)
        #x = self.dropout(feat)
        #x = self.fc(x)
        #if features:
        return feat
        #else:
        #    return x
        
def get_features(model, images):
    model.eval()  
    with torch.no_grad():
        features = model(images, features=True)
    features = features.squeeze(3).squeeze(2).cpu().numpy()
    return features

def feature_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2):
    # https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance
    # HINT: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
    # Implement FID score
    fid = linalg.norm(mu1 - mu2)**2 + (sigma1 + sigma2 - 2*linalg.sqrtm(sigma1@sigma2)).trace()

    return fid

def eval(config = None, pth = None):
    with wandb.init(config=config, 
                    project="ADLCV_final_project",
                    entity="mlops_s194333",):
        print(wandb.config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"Model will run on {device}")
        Data_type = wandb.config.Data

        T = wandb.config.T
        img_size = wandb.config.img_size
        channels = wandb.config.channels
        time_dim = wandb.config.time_dim
        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        num_epochs = wandb.config.num_epochs
        ev = wandb.config.ev
        input_channels=3
        show=False
        beta_start = wandb.config.beta_start
        beta_end = wandb.config.beta_end
        testing = wandb.config.testing
        experiment_name = config.split('/')[-1].split('.')[0]

        model = UNet(img_size=img_size, c_in=2*input_channels, c_out=input_channels, 
                    time_dim=time_dim,channels=channels, device=device).to(device)
        diffusion = Diffusion(img_size=img_size, T=T, beta_start=beta_start, beta_end=beta_end, device=device)
        
        model.load_state_dict(torch.load(pth, map_location=device))
        model.eval()

        vgg = VGG()
        vgg.to(device)
        vgg.eval()
        #vgg.load_state_dict(torch.load('weights/vgg-sprites/model.pth', map_location=device))
        dims = 256 # vgg feature dim

        batch_size = 6

        if Data_type == 'sprites':
            _, _, test_loader = get_Sprites_dataloaders(ev, batch_size, testing=testing)
        elif Data_type == 'exposure':
            test_loader = get_test_dataloader(ev, batch_size, img_size, testing=testing)
        else:
            raise Exception('Data type not supported')

        start_idx = 0

        original_feat = np.empty((len(test_loader.dataset), dims))
        target_feat = np.empty((len(test_loader.dataset), dims))
        generated_feat = np.empty((len(test_loader.dataset), dims))

        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)
            predicted_images = diffusion.p_sample_loop(images, model, batch_size=images.shape[0])
            original_features = get_features(vgg, images)
            target_features = get_features(vgg, target)
            predicted_features = get_features(vgg, predicted_images)
            # store features
            original_feat[start_idx:start_idx + original_features.shape[0]] = original_features
            target_feat[start_idx:start_idx + original_features.shape[0]] = target_features
            generated_feat[start_idx:start_idx + original_features.shape[0]] = predicted_features

            start_idx = start_idx + original_features.shape[0]

            save_images(images=predicted_images, originals=images, targets=target, path=os.path.join("results", experiment_name, f'{start_idx}.jpg'))

        mu_original, sigma_original = feature_statistics(original_feat)
        mu_target, sigma_target = feature_statistics(target_feat)
        mu_pred, sigma_pred = feature_statistics(generated_feat)

        fid = frechet_distance(mu_target, sigma_target, mu_pred, sigma_pred)
        print(f'fid =  {fid:.3f}')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="epoch=19-step=2560.ckpt",
        type=str,
        help="path to ckpt file to evaluate",
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    

    args = parser.parse_args()

    path = args.path
    print(path)
    config = args.config
    print(config)
    eval(config, path)
