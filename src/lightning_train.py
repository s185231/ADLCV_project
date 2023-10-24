import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from src.load_data import get_dataloaders
from src.model import Model


def train(config=None, checkpoint_callbacks=None):
    with wandb.init(config=config, 
                    project="ADLCV_final_project",
                    entity="mlops_s194333",):

        config = wandb.config

        lr = wandb.config.lr
        weight_decay = wandb.config.weight_decay
        epochs = wandb.config.epochs
        batch_size = wandb.config.batch_size
        ev = wandb.config.ev
        image_size = wandb.config.image_size

        device = 0
        model = Model(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )

        wandb.watch(model, log_freq=1)
        logger = pl.loggers.WandbLogger(project="ADLCV_final_project", entity="mlops_s194333")

        trainloader, valloader, _ = get_dataloaders(ev, batch_size, image_size, num_workers=8)

        # make sure no models are saved if no checkpoints are given
        if checkpoint_callbacks is None:
            checkpoint_callbacks = [
                ModelCheckpoint(monitor=False, save_last=False, save_top_k=0)
            ]

        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir="",
            callbacks=checkpoint_callbacks,
            accelerator="gpu",
            devices=[device],
            strategy="ddp",
            logger=logger,
        )

        trainer.fit(
            model=model,
            train_dataloaders=trainloader,
            val_dataloaders=valloader,
        )

        print("Done!")


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(dirpath="models", filename="best")
    train(
        config="src/config/config.yaml",
        checkpoint_callbacks=[checkpoint_callback],
    )
