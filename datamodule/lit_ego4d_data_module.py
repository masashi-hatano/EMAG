from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datamodule.ego4d_dataset import Ego4dHandForecastDataset


class Ego4dHandForecastDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_type,
        test_type,
        data_dir,
        pickle,
        interval,
        interval_ego_input,
        interval_ego_output,
        num_frames_hoi,
        num_frames_ego,
        num_pred_ego,
        mean_rgb,
        std_rgb,
        mean_flow,
        std_flow,
        mean_ego,
        std_ego,
        batch_size,
        num_workers,
        delete,
    ):
        super(Ego4dHandForecastDataModule, self).__init__()
        self.train_type = train_type
        self.test_type = test_type
        self.data_dir = data_dir
        self.pickle_path = Path(pickle, f"interval={interval}")
        self.interval = interval
        self.interval_ego_input = interval_ego_input
        self.interval_ego_output = interval_ego_output
        self.num_frames_hoi = num_frames_hoi
        self.num_frames_ego = num_frames_ego
        self.num_pred_ego = num_pred_ego
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_flow = mean_flow
        self.std_flow = std_flow
        self.mean_ego = mean_ego
        self.std_ego = std_ego
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.delete = delete

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Ego4dHandForecastDataset(
                self.data_dir,
                self.train_type,
                self.pickle_path,
                self.interval,
                self.interval_ego_input,
                self.interval_ego_output,
                self.num_frames_hoi,
                self.num_frames_ego,
                self.num_pred_ego,
                self.mean_rgb,
                self.std_rgb,
                self.mean_flow,
                self.std_flow,
                self.mean_ego,
                self.std_ego,
                self.delete,
            )
            self.val_dataset = Ego4dHandForecastDataset(
                self.data_dir,
                "val",
                self.pickle_path,
                self.interval,
                self.interval_ego_input,
                self.interval_ego_output,
                self.num_frames_hoi,
                self.num_frames_ego,
                self.num_pred_ego,
                self.mean_rgb,
                self.std_rgb,
                self.mean_flow,
                self.std_flow,
                self.mean_ego,
                self.std_ego,
                self.delete,
            )

        if stage == "test" or stage is None:
            self.test_dataset = Ego4dHandForecastDataset(
                self.data_dir,
                self.test_type,
                self.pickle_path,
                self.interval,
                self.interval_ego_input,
                self.interval_ego_output,
                self.num_frames_hoi,
                self.num_frames_ego,
                self.num_pred_ego,
                self.mean_rgb,
                self.std_rgb,
                self.mean_flow,
                self.std_flow,
                self.mean_ego,
                self.std_ego,
                self.delete,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
