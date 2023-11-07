import wandb
import yaml
from src.train import train


if __name__ == "__main__":
    with open("configs/sweep_config_1.yaml", "r") as yamlfile:
        sweep_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="ADLCV_final_project",
        entity="mlops_s194333"
    )
    wandb.agent(sweep_id, train, count=20)
