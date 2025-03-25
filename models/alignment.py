import torch
import torch.nn as nn
from einops import rearrange
from torchvision.ops import roi_align


class Alignment(nn.Module):
    def __init__(self, features_dim=1024, num_frames=8, crop_size=1) -> None:
        super(Alignment, self).__init__()
        self.features_dim = features_dim
        self.num_frames = num_frames
        self.crop_size = crop_size

    def global_avg_pooling(self, features):
        bt, c, h, w = features.shape
        features = nn.AvgPool2d(kernel_size=(h, w))(features)
        features = features.view(-1, c)
        batch_size = int(bt / self.num_frames)
        features = rearrange(features, "(b n t) c -> b n t c", b=batch_size, n=1, t=self.num_frames, c=c)
        return features

    def roi_alignment(self, features, bboxes):
        features_list = []
        bt, _, h, _ = features.shape
        batch_size = int(bt / self.num_frames)
        spatial_scale = h / self.crop_size
        for bbox in bboxes:
            roi_aligned = roi_align(features, [bbox], spatial_scale=spatial_scale, output_size=1)
            # [(B, T), C, H, W] -> [(B, T), C]
            viewed = roi_aligned.view(batch_size * self.num_frames, self.features_dim)
            rearranged = rearrange(
                viewed, "(b n t) c -> b n t c", b=batch_size, n=1, t=self.num_frames
            )
            features_list.append(rearranged)

        # concatenate two features
        features = torch.cat(features_list, dim=1)
        return features

    def forward(self, features, bboxes_obj, bboxes_hand):
        # features: [(B, T), features_dim, 7, 7]
        # bboxes_xxx: [2, (B, T), 4]

        # Avg pooling and hand object roi
        features_global = self.global_avg_pooling(features)
        features_obj = self.roi_alignment(features, bboxes_obj)
        features_hand = self.roi_alignment(features, bboxes_hand)

        return (
            features_global,
            features_obj,
            features_hand,
        )
