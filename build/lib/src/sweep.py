import wandb
import yaml
from src.train import train
import argparse
import torch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sweep_config.yaml')
    args = parser.parse_args()
    config = args.config

    with open(config, "r") as yamlfile:
        sweep_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="ADLCV_final_project",
        entity="mlops_s194333"
    )
    wandb.agent(sweep_id, train, count=20)
