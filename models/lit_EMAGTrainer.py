import math

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange

from netscripts.get_alignment import get_alignment
from netscripts.get_backbone import get_backbone
from netscripts.get_network import get_network
from netscripts.get_optimizer import get_optimizer
from utils.util import scaling_hands, scaling_original


class EMAGTrainer(pl.LightningModule):
    def __init__(
        self, model_config, datamodule_config
    ) -> None:
        super(EMAGTrainer, self).__init__()
        self.backbone_cfg = dict(model_config.backbone)
        self.model_cfg = dict(model_config.transformer)
        self.loss_cfg = dict(model_config.loss)
        self.interval = datamodule_config["interval"]
        self.num_frames = datamodule_config["num_frames_hoi"]
        self.batch_size = datamodule_config["batch_size"]
        self.features_dim = self.model_cfg["src_in_features_hoi"]
        self.optimizer_cfg = dict(model_config.optimizer)
        self.backbone_rgb = get_backbone(**self.backbone_cfg)
        self.backbone_flow = get_backbone(**self.backbone_cfg)
        self.alignment = get_alignment(features_dim=self.features_dim)
        self.model = get_network(**self.model_cfg)

        # initialization
        self.right_list = np.empty(0)
        self.left_list = np.empty(0)
        self.right_final_list = np.empty(0)
        self.left_final_list = np.empty(0)
        self.r_dict = {}
        self.l_dict = {}
        self.right_list = []
        self.left_list = []
        self.right_final_list = []
        self.left_final_list = []

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        dataloader = self.trainer.train_dataloader
        self.iters_per_epoch = math.ceil(len(dataloader.dataset) / self.batch_size)
        optimizer, scheduler = get_optimizer(
            backbone_rgb=self.backbone_rgb,
            backbone_flow=self.backbone_flow,
            model=self.model,
            **self.optimizer_cfg,
            iters_per_epoch=self.iters_per_epoch,
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx: int, metric) -> None:
        for scheduler in self.trainer.lr_schedulers:
            cur_iter = self.trainer.global_step
            next_lr = scheduler["scheduler"].get_epoch_values(cur_iter + 1)[0]
            for param_groups in self.trainer.optimizers[0].param_groups:
                param_groups["lr"] = next_lr

    def get_cosine_weight(self):
        total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
        current_step = self.trainer.global_step
        multiplier = min(1, 0.75 * (1 - math.cos(math.pi * current_step / total_steps)))
        return 1 - multiplier

    def feature_extraction(self, inputs, bboxes, backbone):
        # [B, 4, T, 4] -> [4, (B, T), 4]
        bboxes = rearrange(
            bboxes, "b l t m -> l (b t) m", b=self.B, t=self.num_frames, l=4, m=4
        )
        bboxes_obj = bboxes[:2, :, :]
        bboxes_hand = bboxes[2:4, :, :]
        assert bboxes_obj.shape == (2, self.B * self.num_frames, 4)
        assert bboxes_hand.shape == (2, self.B * self.num_frames, 4)
        # [B, T, C, H, W] -> [(B, T), C, H, W]
        inputs = rearrange(inputs, "b t c h w -> (b t) c h w")

        # feature extraction
        # [(B, T), 3, 224, 224] -> [(B, T), 512, 7, 7]
        features = backbone(inputs)
        # features: [(B, T), 512, 7, 7] -> fg: [B, 1, T, 512]
        # bboxes_obj: [2, (B, T), 4]    -> fo: [B, 2, T, 512]
        # bboxes_hand: [2, (B, T), 4]   -> fh: [B, 2, T, 512]
        fg, fo, fh = self.alignment(features, bboxes_obj, bboxes_hand)
        assert fg.shape == (self.B, 1, self.num_frames, self.features_dim)

        # concat
        feat = torch.cat((fg, fo, fh), dim=1)
        assert feat.shape == (self.B, 5, self.num_frames, self.features_dim)
        return feat

    def training_step(self, batch, batch_idx):
        input, _, meta = batch
        frames = input["frames"]
        flows = input["flows"]
        bboxes = input["bboxes"]
        ego_feat = input["egos_input"]
        attention_mask = input["attention_mask"]
        future_hands = input["future_hands"]
        future_egos = input["future_egos"]
        hands_mask_true = input["hands_mask"]
        egos_mask_true = input["egos_mask"]
        self.B, _, self.T, _ = future_hands.shape
        hands_mask_false = torch.zeros((self.B, 2, self.T)).type_as(frames)
        egos_mask_false = torch.zeros((self.B, self.T)).type_as(frames)

        # Backbone and Alignment
        feat_rgb = self.feature_extraction(frames, bboxes, self.backbone_rgb)
        feat_flow = self.feature_extraction(flows, bboxes, self.backbone_flow)
        feat = torch.cat((feat_rgb, feat_flow), dim=1)
        bbox_feat = bboxes

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        # Transformer
        pred_hands, pred_egos = self.model(
            feat,
            bbox_feat=bbox_feat,
            fmats_input=ego_feat,
            attention_mask=attention_mask,
            future_hands=future_hands,
            future_egos=future_egos,
            hands_mask=hands_mask_false,
            egos_mask=egos_mask_false,
        )
        # mask invalid value
        # past_hands: [B, 2, To, 2]
        # past_egos: [B, 1, To, 2]
        bboxes_hands = (bboxes[:, 2:4, :, :2] + bboxes[:, 2:4, :, 2:]) / 2
        bboxes_hands = scaling_hands(bboxes_hands, meta)

        # mask invalid value
        # pred_hands: [B, 2, T, 2]
        # pred_egos: [B, 1, T, 9]
        pred_hands = pred_hands * hands_mask_true[:, :, :, None]
        pred_hands = scaling_hands(pred_hands, meta)
        future_hands = scaling_hands(future_hands, meta)
        pred_hands = pred_hands[pred_hands != 0]
        future_hands = future_hands[future_hands != 0]
        loss_fun = torch.nn.SmoothL1Loss(reduction="mean", beta=5.0)
        # Compute the loss.
        recon_loss_hand = loss_fun(pred_hands, future_hands)
        pred_egos = pred_egos * egos_mask_true.unsqueeze(1)[:, :, :, None]
        recon_loss_ego = self.train_recon_loss(pred_egos, future_egos.unsqueeze(1))

        loss = recon_loss_hand + recon_loss_ego
        outputs = {
            "loss": loss.item(),
            "recon_loss_hand": recon_loss_hand.item(),
            "recon_loss_ego": recon_loss_ego.item(),
        }
        self.training_step_outputs.append(outputs)
        return loss

    def training_epoch_end(self, outputs):
        recon_loss = np.mean([tmp["loss"] for tmp in self.training_step_outputs])
        recon_loss_hand = np.mean(
            [tmp["recon_loss_hand"] for tmp in self.training_step_outputs]
        )
        recon_loss_ego = np.mean(
            [tmp["recon_loss_ego"] for tmp in self.training_step_outputs]
        )
        self.training_step_outputs.clear()
        self.log("train_recon_loss", recon_loss, on_step=False)
        self.log("train_recon_loss_hand", recon_loss_hand, on_step=False)
        self.log("train_recon_loss_ego", recon_loss_ego, on_step=False)

    def validation_step(self, batch, batch_idx):
        input, _, meta = batch
        frames = input["frames"]
        flows = input["flows"]
        bboxes = input["bboxes"]
        ego_feat = input["egos_input"]
        attention_mask = input["attention_mask"]
        future_hands = input["future_hands"]
        future_egos = input["future_egos"]
        hands_mask_true = input["hands_mask"]
        egos_mask_true = input["egos_mask"]
        self.B, _, self.T, _ = future_hands.shape
        hands_mask_false = torch.zeros((self.B, 2, self.T)).type_as(frames)
        egos_mask_false = torch.zeros((self.B, self.T)).type_as(frames)

        # Backbone and Alignment
        feat_rgb = self.feature_extraction(frames, bboxes, self.backbone_rgb)
        feat_flow = self.feature_extraction(flows, bboxes, self.backbone_flow)
        feat = torch.cat((feat_rgb, feat_flow), dim=1)
        bbox_feat = bboxes

        # HOI Ego Transformer
        pred_hands, pred_egos = self.model(
            feat,
            bbox_feat=bbox_feat,
            fmats_input=ego_feat,
            attention_mask=attention_mask,
            future_hands=future_hands,
            future_egos=future_egos,
            hands_mask=hands_mask_false,
            egos_mask=egos_mask_false,
        )

        # adjust the scale and mask invalid value
        pred_hands = scaling_hands(pred_hands, meta)
        future_hands = scaling_hands(future_hands, meta)
        pred_hands = pred_hands * hands_mask_true[:, :, :, None]
        pred_hands = pred_hands[pred_hands != 0]
        future_hands = future_hands[future_hands != 0]
        loss_fun = torch.nn.SmoothL1Loss(reduction="mean", beta=5.0)
        # Compute the loss.
        recon_loss_hand = loss_fun(pred_hands, future_hands)
        pred_egos = pred_egos * egos_mask_true.unsqueeze(1)[:, :, :, None]
        recon_loss_ego = self.validation_recon_loss(pred_egos, future_egos.unsqueeze(1))

        loss = recon_loss_hand + recon_loss_ego
        outputs = {
            "loss": loss.item(),
            "recon_loss_hand": recon_loss_hand.item(),
            "recon_loss_ego": recon_loss_ego.item(),
        }
        self.validation_step_outputs.append(outputs)
        return loss

    def validation_epoch_end(self, outputs):
        total_loss = np.mean([tmp["loss"] for tmp in self.validation_step_outputs])
        recon_loss_hand = np.mean(
            [tmp["recon_loss_hand"] for tmp in self.validation_step_outputs]
        )
        recon_loss_ego = np.mean(
            [tmp["recon_loss_ego"] for tmp in self.validation_step_outputs]
        )
        self.validation_step_outputs.clear()
        self.log("val_total_loss", total_loss, on_step=False)
        self.log("val_recon_loss", recon_loss_hand, on_step=False)
        self.log("val_recon_loss_ego", recon_loss_ego, on_step=False)

    def test_step(self, batch, batch_idx):
        input, _, meta = batch
        frames = input["frames"]
        flows = input["flows"]
        bboxes = input["bboxes"]
        ego_feat = input["egos_input"]
        attention_mask = input["attention_mask"]
        future_hands = input["future_hands"]
        future_egos = input["future_egos"]
        hands_mask_true = input["hands_mask"]
        # egos_mask_true = input["egos_mask"]
        # ref_pos = input["ref_pos"]
        # bc = input["bc"]
        self.B, _, self.T, _ = future_hands.shape
        hands_mask_false = torch.zeros((self.B, 2, self.T)).type_as(frames)
        egos_mask_false = torch.zeros((self.B, self.T)).type_as(frames)

        # Backbone and Alignment
        feat_rgb = self.feature_extraction(frames, bboxes, self.backbone_rgb)
        feat_flow = self.feature_extraction(flows, bboxes, self.backbone_flow)
        feat = torch.cat((feat_rgb, feat_flow), dim=1)
        # feat = feat_rgb
        # bbox_feat = torch.cat((bboxes, bboxes), dim=1)
        bbox_feat = bboxes
        # bbox_feat = bboxes_rel

        # HOI Ego Transformer
        pred_hands, pred_egos = self.model(
            feat,
            bbox_feat=bbox_feat,
            fmats_input=ego_feat,
            attention_mask=attention_mask,
            future_hands=future_hands,
            future_egos=future_egos,
            hands_mask=hands_mask_false,
            egos_mask=egos_mask_false,
        )

        # [B, 2, 4, 2] -> [B, 16]
        pred_hands = rearrange(
            pred_hands, "b n t m -> b (t n m)", b=self.B, n=2, t=4, m=2
        )
        future_hands = rearrange(
            future_hands, "b n t m -> b (t n m)", b=self.B, n=2, t=4, m=2
        )
        hands_mask_true = torch.cat(
            (hands_mask_true[:, :, :, None], hands_mask_true[:, :, :, None]), dim=-1
        )
        hands_mask_true = rearrange(
            hands_mask_true, "b n t m -> b (t n m)", b=self.B, n=2, t=4, m=2
        )

        # adjust the scale and mask invalid value
        # pred_hands = pred_hands * hands_mask_true[:, :, :, None]
        pred_hands = torch.mul(pred_hands, hands_mask_true)
        future_hands = torch.mul(future_hands, hands_mask_true)
        pred_hands = scaling_original(pred_hands, meta)
        future_hands = scaling_original(future_hands, meta)

        # calculate L2 distance
        self.calculate_L2_loss(
            pred_hands.cpu(),
            future_hands.cpu(),
        )

    def test_epoch_end(self, outputs: dict) -> None:
        lmean_disp = np.mean(self.left_list)
        rmean_disp = np.mean(self.right_list)
        lcontact_disp = np.mean(self.left_final_list)
        rcontact_disp = np.mean(self.right_final_list)
        self.log("left hand mean disp error", lmean_disp, on_step=False)
        self.log("right hand mean disp error", rmean_disp, on_step=False)
        self.log("left hand contact disp error", lcontact_disp, on_step=False)
        self.log("right hand contact disp error", rcontact_disp, on_step=False)

    def train_total_loss(self, traj_loss, traj_kl_loss, motion_loss, motion_kl_loss):
        losses = {}
        total_loss = 0
        lambda_traj = self.loss_cfg["lambda_traj"]
        lambda_traj_kl = self.loss_cfg["lambda_traj_kl"]
        lambda_motion = self.loss_cfg["lambda_motion"]
        lambda_motion_kl = self.loss_cfg["lambda_motion_kl"]
        if lambda_traj is not None and traj_loss is not None:
            total_loss += lambda_traj * traj_loss.sum()
            losses["traj_loss"] = traj_loss.detach()
        else:
            losses["traj_loss"] = 0.0

        if lambda_traj_kl is not None and traj_kl_loss is not None:
            total_loss += lambda_traj_kl * traj_kl_loss.sum()
            losses["traj_kl_loss"] = traj_kl_loss.detach()
        else:
            losses["traj_kl_loss"] = 0.0

        if lambda_motion is not None and motion_loss is not None:
            total_loss += lambda_motion * motion_loss.sum()
            losses["motion_loss"] = motion_loss.detach()
        else:
            losses["motion_loss"] = 0.0

        if lambda_motion_kl is not None and motion_kl_loss is not None:
            total_loss += lambda_motion_kl * motion_kl_loss.sum()
            losses["motion_kl_loss"] = motion_kl_loss.detach()
        else:
            losses["motion_kl_loss"] = 0.0

        if total_loss is not None:
            losses["loss"] = total_loss
        else:
            losses["loss"] = 0.0
        return losses

    def train_recon_loss(self, preds, targets):
        b, n, t, m = preds.shape
        preds = rearrange(preds, "b n t m -> (b n t) m", b=b, n=n, t=t, m=m)
        targets = rearrange(targets, "b n t m -> (b n t) m", b=b, n=n, t=t, m=m)
        recon_loss = torch.sum((preds - targets) ** 2, dim=1)
        recon_loss = recon_loss[recon_loss != 0].mean()
        return recon_loss

    def validation_recon_loss(self, preds, targets):
        b, n, t, m = preds.shape
        preds = rearrange(preds, "b n t m -> (b n t) m", b=b, n=n, t=t, m=m)
        targets = rearrange(targets, "b n t m -> (b n t) m", b=b, n=n, t=t, m=m)
        recon_loss = torch.sum((preds - targets) ** 2, dim=1)
        recon_loss = recon_loss[recon_loss != 0].mean()
        return recon_loss

    def calculate_L2_loss(self, preds, labels):
        for pred, label in zip(preds, labels):
            for k in range(4):
                l_x_pred = pred[k * 4]
                l_y_pred = pred[k * 4 + 1]
                r_x_pred = pred[k * 4 + 2]
                r_y_pred = pred[k * 4 + 3]

                l_x_gt = label[k * 4]
                l_y_gt = label[k * 4 + 1]
                r_x_gt = label[k * 4 + 2]
                r_y_gt = label[k * 4 + 3]

                if r_x_gt != 0 or r_y_gt != 0:
                    dist = np.sqrt((r_y_gt - r_y_pred) ** 2 + (r_x_gt - r_x_pred) ** 2)
                    self.right_list.append(dist)
                if l_x_gt != 0 or l_y_gt != 0:
                    dist = np.sqrt((l_y_gt - l_y_pred) ** 2 + (l_x_gt - l_x_pred) ** 2)
                    self.left_list.append(dist)

            if r_x_gt != 0 or r_y_gt != 0:
                dist = np.sqrt((r_y_gt - r_y_pred) ** 2 + (r_x_gt - r_x_pred) ** 2)
                self.right_final_list.append(dist)
            if l_x_gt != 0 or l_y_gt != 0:
                dist = np.sqrt((l_y_gt - l_y_pred) ** 2 + (l_x_gt - l_x_pred) ** 2)
                self.left_final_list.append(dist)
