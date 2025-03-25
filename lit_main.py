import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule.lit_ego4d_data_module import Ego4dHandForecastDataModule
from models.lit_EMAGTrainer import EMAGTrainer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config_ego4d.yaml")
def main(config):
    # configs
    cfg = config.main
    model_config = config.model_config
    datamodule_config = dict(config.data_module)

    # initialize random seeds
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # data module
    data_module = Ego4dHandForecastDataModule(**datamodule_config)

    model = EMAGTrainer(
        model_config,
        datamodule_config,
    )

    if torch.cuda.is_available() and len(cfg.devices):
        print(f"Using {len(cfg.devices)} GPUs !")

    train_logger = loggers.TensorBoardLogger("tensor_board", default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_recon_loss",
        dirpath="checkpoints/",
        filename="{epoch:02d}-{val_recon_loss:.4f}",
        save_top_k=5,
        mode="min",
    )

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        # strategy=cfg.strategy,
        max_epochs=model_config.optimizer.max_epochs,
        logger=train_logger,
        callbacks=[checkpoint_callback],
        detect_anomaly=True,
    )

    if cfg.train:
        if cfg.resume_ckpt is not None:
            trainer.fit(model, data_module, ckpt_path=cfg.resume_ckpt)
        else:
            trainer.fit(model, data_module)
        print(trainer.callback_metrics)
    if cfg.test:
        logging.basicConfig(level=logging.DEBUG)
        model = EMAGTrainer.load_from_checkpoint(
            model_config=model_config,
            datamodule_config=datamodule_config,
            checkpoint_path=cfg.ckpt_pth,
        )
        trainer.test(model, data_module)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
