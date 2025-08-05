import torch.nn as nn
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
import argparse
import yaml
import torch
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loss_logger_callback import LossLoggerCallback

from datamodule import PathIntegrationDataModule

import os

from model import PathIntRNN
from model_lightning import PathIntRNNLightning

import datetime

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
print("Log directory:", log_dir)


def main(config: dict):
    # Generate unique run identifier
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting training run: {run_id}")

    wandb.init(
        project=config["project_name"],
        name=f"{config['project_name']}_{run_id}",
        dir=log_dir,
    )

    print("Wandb initialized. Find logs at: ", log_dir)
    print(f"Wandb run name: {config['project_name']}_{run_id}")

    wandb_logger = WandbLogger(
        name=f"{config['project_name']}_{run_id}",
        dir=log_dir
    )

    datamodule = PathIntegrationDataModule(
        num_trajectories=config["num_trajectories"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        train_val_split=config["train_val_split"],
        start_time=config["start_time"],
        end_time=config["end_time"],
        num_time_steps=config["num_time_steps"],
        arena_L=config["arena_L"],
        mu_speed=config["mu_speed"],
        sigma_speed=config["sigma_speed"],
        tau_vel=config["tau_vel"],
    )

    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print("Data prepared")




    model = PathIntRNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        alpha=config["alpha"],
    )

    print("PathIntRNN initialized")

    rnn_lightning = PathIntRNNLightning(
        model=model,
        lr=config["learning_rate"],
    )

    rnn_lightning.to(config["device"])

    print("RNNLightning initialized")

    run_dir = os.path.join(log_dir, "checkpoints", run_id)
    
    @rank_zero_only
    def create_directories():
        os.makedirs(run_dir, exist_ok=True)
    
    create_directories()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir, 
        filename="best-model-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    
    loss_logger = LossLoggerCallback(save_dir=run_dir)

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, loss_logger],
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
    )

    print("Trainer initialized")
    print("Training...")

    trainer.fit(rnn_lightning, train_loader, val_loader)

    print("Training complete!")
    
    # Only save on rank 0
    @rank_zero_only
    def save_additional_artifacts():
        model_path = os.path.join(run_dir, f"final_model_{run_id}.pth")
        torch.save(rnn_lightning.model.state_dict(), model_path)
        
        config_path = os.path.join(run_dir, f"config_{run_id}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        print(f"All artifacts saved to: {run_dir}")
    
    save_additional_artifacts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RNN training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path containing the config file",
    )
    args = parser.parse_args()

    config_path = args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    main(config)
