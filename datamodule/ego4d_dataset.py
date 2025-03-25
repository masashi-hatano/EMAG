import json
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class Ego4dHandForecastDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        mode,
        pickle_path,
        interval,
        interval_ego_input,
        interval_ego_label,
        num_frames_hoi,
        num_frames_ego,
        num_pred_ego,
        mean_rgb,
        std_rgb,
        mean_flow,
        std_flow,
        mean_ego,
        std_ego,
        delete,
    ):
        super(Ego4dHandForecastDataset, self).__init__()
        self.data_dir = data_dir
        # Only support train, val, test, and trainval mode.
        assert mode in [
            "train",
            "val",
            "test",
            "trainval",
        ], f"Split '{mode}' not supported for Ego4D Hand Anticipation"
        self.mode = mode
        self.path_to_json = Path(self.data_dir, f"annotations/fho_hands_{mode}.json")
        self.path_to_ego = Path(self.data_dir, f"ego-motion/ego_{mode}.json")
        self.path_to_pickle = pickle_path
        self.interval = interval
        self.interval_ego_input = interval_ego_input
        self.interval_ego_label = interval_ego_label
        self.num_frames_hoi = num_frames_hoi
        self.num_frames_ego = num_frames_ego
        self.num_pred_ego = num_pred_ego
        self.transforms_rgb = transforms.Normalize(mean_rgb, std_rgb)
        self.transforms_flow = transforms.Normalize(mean_flow, std_flow)
        self.transforms_fmat = transforms.Normalize(mean_ego, std_ego)
        self.delete = delete
        self._construct_loader()

    def normalize(self, label, h, w):
        mask = torch.cat((torch.zeros((2, 5, 1)), torch.ones((2, 5, 1))), dim=2)
        label = label * mask / h + label * (1 - mask) / w
        return label

    def _construct_loader(self):
        # initialization
        self._left_hand_visibility = []
        self._right_hand_visibility = []
        self._path_to_img_frame = []
        self._clip_uid = []
        self._pre45_clip_frames = []
        self._pre45_frames = []
        self._labels = []
        self._labels_masks = []
        self._clip_idx = []
        self._spatial_temporal_idx = []

        with open(self.path_to_json) as f:
            data = json.load(f)
        for clip_dict in data["clips"]:
            video_uid = clip_dict["video_uid"]
            if video_uid in self.delete:
                print(
                    f"{video_uid} is invalid video, so it will not be included in the dataset"
                )
                continue
            for frame_dict in clip_dict["frames"]:
                pre45_clip_frame = frame_dict["pre_45"]["clip_frame"]
                if pre45_clip_frame <= self.interval * self.num_frames_hoi:
                    continue
                clip_uid = clip_dict["clip_uid"]

                # placeholder for the 2x5x2 hand gt vector (padd zero when GT is not available)
                # 5 frames have the following order: pre_45, pre_40, pre_15, pre, contact
                # GT for each frames has the following order: left_x,left_y,right_x,right_y
                label = torch.zeros((2, 5, 2))
                label_mask = torch.zeros((2, 5))
                if self.mode in ["train", "val", "trainval"]:
                    for frame_type, frame_annot in frame_dict.items():
                        if frame_type in [
                            "action_start_sec",
                            "action_end_sec",
                            "action_start_frame",
                            "action_end_frame",
                            "action_clip_start_sec",
                            "action_clip_end_sec",
                            "action_clip_start_frame",
                            "action_clip_end_frame",
                        ]:
                            continue
                        if frame_type == "pre_45":
                            for single_hand in frame_annot["boxes"]:
                                if "left_hand" in single_hand:
                                    label_mask[0][0] = 1.0
                                    label[0][0] = torch.tensor(single_hand["left_hand"])
                                if "right_hand" in single_hand:
                                    label_mask[1][0] = 1.0
                                    label[1][0] = torch.tensor(
                                        single_hand["right_hand"]
                                    )
                        if frame_type == "pre_30":
                            for single_hand in frame_annot["boxes"]:
                                if "left_hand" in single_hand:
                                    label_mask[0][1] = 1.0
                                    label[0][1] = torch.tensor(single_hand["left_hand"])
                                if "right_hand" in single_hand:
                                    label_mask[1][1] = 1.0
                                    label[1][1] = torch.tensor(
                                        single_hand["right_hand"]
                                    )
                        if frame_type == "pre_15":
                            for single_hand in frame_annot["boxes"]:
                                if "left_hand" in single_hand:
                                    label_mask[0][2] = 1.0
                                    label[0][2] = torch.tensor(single_hand["left_hand"])
                                if "right_hand" in single_hand:
                                    label_mask[1][2] = 1.0
                                    label[1][2] = torch.tensor(
                                        single_hand["right_hand"]
                                    )
                        if frame_type == "pre_frame":
                            for single_hand in frame_annot["boxes"]:
                                if "left_hand" in single_hand:
                                    label_mask[0][3] = 1.0
                                    label[0][3] = torch.tensor(single_hand["left_hand"])
                                if "right_hand" in single_hand:
                                    label_mask[1][3] = 1.0
                                    label[1][3] = torch.tensor(
                                        single_hand["right_hand"]
                                    )
                        if frame_type == "contact_frame":
                            for single_hand in frame_annot["boxes"]:
                                if "left_hand" in single_hand:
                                    label_mask[0][4] = 1.0
                                    label[0][4] = torch.tensor(single_hand["left_hand"])
                                if "right_hand" in single_hand:
                                    label_mask[1][4] = 1.0
                                    label[1][4] = torch.tensor(
                                        single_hand["right_hand"]
                                    )
                # label mask should be a non-zero tensor
                if torch.count_nonzero(label_mask) == 0:
                    continue

                path_to_img_frame = Path(self.data_dir, "image_frame", clip_uid)
                self._clip_idx.append(clip_uid)
                self._path_to_img_frame.append(path_to_img_frame)
                self._clip_uid.append(clip_uid)
                self._pre45_clip_frames.append(pre45_clip_frame)
                self._pre45_frames.append(frame_dict["pre_45"]["frame"])
                self._labels.append(label)
                self._labels_masks.append(label_mask)

        logger.info(
            "Constructing Ego4D dataloader (size: {})".format(
                len(self._pre45_clip_frames)
            )
        )

    def _get_input(
        self, pre45_clip_frame, input_dir_rgb, input_dir_flow, input_dir_fmat
    ):
        # prepare frame_names
        frame_names_vis = list(
            reversed(
                [
                    max(1, pre45_clip_frame - self.interval * i)
                    for i in range(1, self.num_frames_hoi + 1)
                ]
            )
        )
        # initialization
        frames = torch.zeros(self.num_frames_hoi, 224, 224, 3)
        flows = torch.zeros(self.num_frames_hoi, 224, 224, 2)
        flow_dummy = torch.zeros(self.num_frames_hoi, 224, 224, 1)
        fmats_input = torch.zeros(self.num_frames_ego, 3, 3)
        fmats_label = torch.zeros(self.num_pred_ego, 3, 3)
        fmats_label_mask = torch.zeros(self.num_pred_ego)
        # loop for visual information
        for i, frame in enumerate(frame_names_vis):
            input_path_rgb = input_dir_rgb / Path(str(frame).zfill(6))
            input_path_flow = input_dir_flow / Path("npy") / Path(str(frame).zfill(6))
            img = cv2.imread(str(input_path_rgb) + ".png").astype(np.float32)
            opt = np.load(str(input_path_flow) + ".npy").astype(np.float32)
            frames[i] = torch.from_numpy(cv2.resize(img, (224, 224)))
            flows[i] = torch.from_numpy(opt)
        flows = torch.cat((flows, flow_dummy), dim=-1)
        # loop for ego motion (fundamental matrix)
        for j in range(self.num_frames_ego + self.num_pred_ego):
            if j < self.num_frames_ego:
                frame = max(1, pre45_clip_frame - self.interval_ego_input * (self.num_frames_ego - j))
                input_path_fmat = input_dir_fmat / Path(str(frame).zfill(6) + ".npy")
                fmat = np.load(str(input_path_fmat)).astype(np.float32)
                fmats_input[j] = torch.from_numpy(fmat)
            else:
                frame = pre45_clip_frame + self.interval_ego_label * (
                    j - self.num_frames_ego
                )
                input_path_fmat = input_dir_fmat / Path(str(frame).zfill(6) + ".npy")
                if not input_path_fmat.exists():
                    fmat = np.zeros((3, 3))
                else:
                    fmat = np.load(str(input_path_fmat)).astype(np.float32)
                    fmats_label_mask[j - self.num_frames_ego] = 1
                fmats_label[j - self.num_frames_ego] = torch.from_numpy(fmat)

        # [T, H, W, C] -> [T, C, H, W]
        frames = frames.permute(0, 3, 1, 2)
        flows = flows.permute(0, 3, 1, 2)

        # Perform normalization
        frames = self.transforms_rgb(frames / 255.0)
        flows = self.transforms_flow(flows)
        fmats_input = self.transforms_fmat(
            fmats_input.view(self.num_frames_ego, 9, 1, 1)
        )
        fmats_label = self.transforms_fmat(fmats_label.view(self.num_pred_ego, 9, 1, 1))
        fmats_input = fmats_input.view(self.num_frames_ego, 9)
        fmats_label = fmats_label.view(self.num_pred_ego, 9)
        return frames, flows, fmats_input, fmats_label, fmats_label_mask

    def __getitem__(self, index):
        input = {}
        clip_uid = self._clip_uid[index]
        pre45_clip_frame = self._pre45_clip_frames[index]
        input_dir_rgb = self._path_to_img_frame[index]
        input_dir_flow = Path(str(input_dir_rgb).replace("image_frame", "optical_flow"))
        input_dir_fmat = Path(str(input_dir_rgb).replace("image_frame", "fundamental"))
        pickle_path = self.path_to_pickle / Path(
            clip_uid, str(pre45_clip_frame) + ".pkl"
        )

        # load frames
        frames, flows, fmats_input, fmats_label, fmats_label_mask = self._get_input(
            pre45_clip_frame, input_dir_rgb, input_dir_flow, input_dir_fmat
        )

        # load bboxes from pickle
        with open(pickle_path, "rb") as f:
            p = pickle.load(f)
        bo = p["bboxes_obj"]
        bh = p["bboxes_hand"]
        attention_mask = p["attention_mask"]

        # meta info
        input_path_rgb = input_dir_rgb / Path(str(pre45_clip_frame).zfill(6))
        img = cv2.imread(str(input_path_rgb) + ".png")
        h, w, _ = img.shape

        label = self._labels[index]
        label = torch.FloatTensor(label)
        label = self.normalize(label, h, w)
        hands_mask = self._labels_masks[index]
        hands_mask = torch.FloatTensor(hands_mask)

        input["frames"] = frames
        input["flows"] = flows
        input["bboxes"] = torch.cat((bo, bh), dim=0)
        input["ego_feat"] = fmats_input
        input["attention_mask"] = attention_mask
        input["future_hands"] = label
        input["future_egos"] = fmats_label
        input["hands_mask"] = hands_mask
        input["egos_mask"] = fmats_label_mask
        idx = (self._clip_idx[index], self._pre45_frames[index] - 1)
        meta = [str(input_dir_rgb), pre45_clip_frame, h, w, idx]

        return input, index, meta

    def __len__(self):
        return len(self._path_to_img_frame)
