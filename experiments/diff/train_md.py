import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

import sys
from argparse import ArgumentParser
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import multiprocessing
# multiprocessing.set_start_method('fork')
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
import setproctitle
import yaml

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from motion_diff.dataset.data_module import DataModule
from motion_diff.model import MotionDiff


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--logger", type=str, default='none', choices=['none', 'qualcomm', 'wandb', 'tensorboard'])
    parser.add_argument("--viz", action='store_true')
    parser.add_argument("--viz_interv", type=int, default=400)
    args = parser.parse_args()
    # args = parser.parse_args('--config motion_diff/configs/vi_config_pca.yml --save tmp --exp_name debug --logger none'.split(' '))
    # args = parser.parse_args('--config motion_diff/configs/vi_config_m2minter.yml --save tmp --exp_name debug --logger tensorboard --viz'.split(' '))
    return args


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["exp_name"] = args.exp_name
    pl.seed_everything(cfg["seed"], workers=True)
    setproctitle.setproctitle(cfg["title"])

    torch.set_float32_matmul_precision("high")
    model = MotionDiff(cfg)
    datamodule = DataModule(cfg)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_checkpoint = ModelCheckpoint(monitor='val_overlap_rate', save_top_k=10, mode='min')
    model.viz = args.viz
    model.viz_interv = args.viz_interv

    if args.logger == 'qualcomm':
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        apikey = 'local-59bb96d6f8d5016c2feab1953b3ec8451858bd8d'
        host = 'https://server.auto-wandb.qualcomm.com/'
        wandb.login(key=apikey, host=host)
        logger = WandbLogger(project='LCSim', save_dir=args.save, name=cfg["exp_name"])
    elif args.logger == 'wandb':
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        wandb.login(key='cf40207594485117b279359ef28e19da3cf02fbf')
        logger = WandbLogger(project='LCSim', save_dir=args.save, name=cfg["exp_name"])
    elif args.logger == 'tensorboard':
        from pytorch_lightning import loggers as pl_loggers
        logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save, name=cfg["exp_name"]
        )
    elif args.logger == 'none':
        logger = None
        

    trainer = pl.Trainer(
        logger=logger,
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
        callbacks=[lr_monitor, model_checkpoint],
        max_epochs=cfg["trainer"]["max_epochs"],
        check_val_every_n_epoch=cfg["trainer"]["val_interval"],
    )
    trainer.fit(model, datamodule, ckpt_path=cfg["trainer"]["ckpt_path"])
    # trainer.validate(model, datamodule, ckpt_path=cfg["trainer"]["ckpt_path"])


if __name__ == "__main__":
    main()
