import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from torch import optim
import logging
import datetime
import argparse
import yaml
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

from ddpm import Diffusion
from model import UNet

import wandb
from src.load_data import get_dataloaders


def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()


def create_result_folders(experiment_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)

def train(config = None, device = None):
    with wandb.init(config=config, 
                    project="ADLCV_final_project",
                    entity="mlops_s194333",):

        print(wandb.config)

        T = wandb.config.T
        img_size = wandb.config.img_size
        channels = wandb.config.channels
        time_dim = wandb.config.time_dim
        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        num_epochs = wandb.config.num_epochs
        ev = wandb.config.ev
        # split config_path
        experiment_name = config.split('/')[-1].split('.')[0]
        input_channels=3
        show=False
        beta_start = wandb.config.beta_start
        beta_end = wandb.config.beta_end


        """Implements algrorithm 1 (Training) from the ddpm paper at page 4"""
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        create_result_folders(os.path.join(experiment_name, time_stamp))
        trainloader, valloader, _ = get_dataloaders(ev, batch_size, img_size)

        model = UNet(img_size=img_size, c_in=2*input_channels, c_out=input_channels, 
                    time_dim=time_dim,channels=channels, device=device).to(device)
        diffusion = Diffusion(img_size=img_size, T=T, beta_start=beta_start, beta_end=beta_end, device=device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        # use MSE loss
        mse = torch.nn.MSELoss()

        logger = SummaryWriter(os.path.join("runs", experiment_name, time_stamp))
        l = len(trainloader)

        for epoch in range(1, num_epochs + 1):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(trainloader)

            for i, (image, target) in enumerate(pbar):
                image = image.to(device)
                target = target.to(device)

                # TASK 4: implement the training loop
                t = diffusion.sample_timesteps(image.shape[0]).to(device) # line 3 from the Training algorithm
                x_t, noise = diffusion.q_sample(target, t) # inject noise to the images (forward process), HINT: use q_sample
                # concatenate x_t and image
                x_t = torch.cat((x_t, image), dim=1)
                predicted_noise = model(x_t, t) # predict noise of x_t using the UNet
                noise_loss = mse(noise, predicted_noise) # loss between noise and predicted noise
                loss = noise_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

            # time stamp
            
            sampled_images = diffusion.p_sample_loop(image, model, batch_size=image.shape[0])
            save_images(images=sampled_images, path=os.path.join("results", experiment_name, time_stamp, f"pred_{epoch}.jpg"),
                        show=show, title=f'Epoch {epoch}')
            save_images(images=image, path=os.path.join("results", experiment_name, time_stamp, f"true_{epoch}.jpg"),
                        show=show, title=f'Epoch {epoch}')
            
            torch.save(model.state_dict(), os.path.join("models", experiment_name, time_stamp, f"weights-{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    config = args.config
    print(config)
    train(config, device)


        