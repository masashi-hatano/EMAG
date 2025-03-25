import torch
import torch.nn as nn
from einops import rearrange

from models.embedding import (Decoder_PositionalEmbedding,
                              Encoder_PositionalEmbedding, PositionalEncoding)
from models.layer import DecoderBlock, EncoderBlock
from models.net_utils import get_pad_mask, get_subsequent_mask, trunc_normal_


class Encoder(nn.Module):
    def __init__(
        self,
        num_patches=5,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        dropout=0.0,
        time_embed_type=None,
        num_frames=None,
    ):
        super().__init__()
        if time_embed_type is None or num_frames is None:
            time_embed_type = "sin"
        self.time_embed_type = time_embed_type
        self.num_patches = (
            num_patches  # (hand, object global feature patches, default: 5)
        )
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Encoder_PositionalEmbedding(embed_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(embed_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "time_embed"}

    def forward(self, x, mask=None, ref_pos=None):
        B, T, N = x.shape[:3]

        x = rearrange(x, "b t n m -> (b t) n m", b=B, t=T, n=N)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
        x = self.time_embed(x)
        x = self.time_drop(x)
        x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=T)

        mask = mask.transpose(1, 2)
        for blk in self.encoder_blocks:
            x = blk(x, B, T, N, mask=mask, ref_pos=ref_pos)

        x = rearrange(x, "b (n t) m -> b t n m", b=B, t=T, n=N)
        x = self.norm(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        in_features,
        num_patches,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        dropout=0.0,
        time_embed_type=None,
        num_frames=None,
    ):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.trg_embedding = nn.Linear(in_features, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        if time_embed_type is None or num_frames is None:
            time_embed_type = "sin"
        self.time_embed_type = time_embed_type
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Decoder_PositionalEmbedding(embed_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(embed_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "time_embed"}

    def forward(self, trg, memory, memory_mask=None, trg_mask=None, ref_pos=None):
        trg = self.trg_embedding(trg)

        B, T, N = trg.shape[:3]

        trg = rearrange(trg, "b t n m -> (b t) n m", b=B, t=T, n=N)
        trg = trg + self.pos_embed

        trg = rearrange(trg, "(b t) n m -> (b n) t m", b=B, t=T)
        trg = self.time_embed(trg)

        trg = rearrange(trg, "(b n) t m -> b (n t) m", b=B, t=T)

        for blk in self.decoder_blocks:
            trg = blk(trg, memory, memory_mask=memory_mask, trg_mask=trg_mask, ref_pos=ref_pos)

        trg = self.norm(trg)
        return trg


class Decoder(nn.Module):
    def __init__(
        self,
        in_features,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        dropout=0.0,
        time_embed_type=None,
        num_frames=None,
    ):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.trg_embedding = nn.Linear(in_features, embed_dim)

        if time_embed_type is None or num_frames is None:
            time_embed_type = "sin"
        self.time_embed_type = time_embed_type
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Decoder_PositionalEmbedding(embed_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(embed_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "time_embed"}

    def forward(self, trg, memory, memory_mask=None, trg_mask=None, ref_pos=None):
        trg = self.trg_embedding(trg)
        trg = self.time_embed(trg)
        trg = self.time_drop(trg)

        for blk in self.decoder_blocks:
            trg = blk(trg, memory, memory_mask=memory_mask, trg_mask=trg_mask, ref_pos=ref_pos)

        trg = self.norm(trg)
        return trg


class EMAG(nn.Module):
    def __init__(
        self,
        src_in_features_hoi,
        trg_in_features_hoi,
        src_in_features_ego,
        trg_in_features_ego,
        input_modality,
        use_flow,
        num_patches_hoi,
        num_patches_ego,
        num_patches_global,
        embed_dim_hoi=512,
        embed_dim_ego=64,
        coord_dim=64,
        num_heads=8,
        enc_depth=6,
        dec_depth=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        dropout=0.0,
        encoder_time_embed_type="sin",
        decoder_time_embed_type="sin",
        num_frames_input_hoi=None,
        num_frames_output_hoi=None,
        num_frames_input_ego=None,
        num_frames_output_ego=None,
    ):
        super().__init__()
        self.input_modality = input_modality
        self.use_flow = use_flow
        self.num_patches_hoi = num_patches_hoi
        self.num_patches_ego = num_patches_ego
        self.num_patches_global = num_patches_global
        self.embed_dim_hoi = embed_dim_hoi
        self.embed_dim_ego = embed_dim_ego
        self.embed_dim_global = embed_dim_hoi
        self.coord_dim = coord_dim
        self.downproject = nn.Linear(src_in_features_hoi, embed_dim_hoi)

        self.bbox_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(self.coord_dim // 2, self.coord_dim),
            nn.ELU(),
        )

        self.bbox_to_feature_direct = nn.Sequential(
            nn.Linear(4, self.coord_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.coord_dim, self.embed_dim_hoi),
            nn.ELU(),
        )

        self.feat_fusion = nn.Sequential(
            nn.Linear(self.embed_dim_hoi + self.coord_dim, self.embed_dim_hoi),
            nn.ELU(inplace=True),
        )

        self.fmat_to_feature = nn.Sequential(
            nn.Linear(src_in_features_ego, self.coord_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.coord_dim, self.embed_dim_ego),
            nn.ELU(),
        )
        self.learnable_token_ego = nn.Parameter(torch.zeros(1, 1, 512))
        self.learnable_token_hand = nn.Parameter(torch.zeros(1, 1, 512))

        self.encoder_hoi = Encoder(
            num_patches=7,
            embed_dim=embed_dim_hoi,
            depth=enc_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            dropout=dropout,
            time_embed_type=encoder_time_embed_type,
            num_frames=num_frames_input_hoi,
        )

        self.decoder_hoi = Decoder(
            in_features=trg_in_features_hoi,
            embed_dim=self.embed_dim_global,
            depth=dec_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            dropout=dropout,
            time_embed_type=decoder_time_embed_type,
            num_frames=num_frames_output_hoi,
        )

        self.decoder_ego = Decoder(
            in_features=trg_in_features_ego,
            embed_dim=self.embed_dim_global,
            depth=dec_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            dropout=dropout,
            time_embed_type=decoder_time_embed_type,
            num_frames=num_frames_output_ego,
        )

        self.hand_head = nn.Sequential(
            nn.Linear(embed_dim_hoi, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )
        self.ego_head = nn.Sequential(
            nn.Linear(embed_dim_ego, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 9)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encoder_input_hoi_only(self, bbox_feat):
        assert self.num_patches_hoi == 4
        return self._encoder_input_hoi_only(bbox_feat)

    def _encoder_input_hoi_only(self, bbox_feat):
        B, _, T, _ = bbox_feat.shape
        bbox_feat = bbox_feat.view(-1, 4)
        bbox_feat = self.bbox_to_feature_direct(bbox_feat)
        bbox_feat = bbox_feat.view(B, -1, T, self.embed_dim_hoi)
        bbox_feat = bbox_feat.transpose(1, 2)
        assert bbox_feat.shape[2] == 4
        return bbox_feat

    def encoder_input_global(self, feat):
        assert self.num_patches_global == 2
        return self._encoder_input_global(feat)

    def _encoder_input_global(self, feat):
        B, _, T, c = feat.shape
        if c != self.embed_dim_global:
            feat = self.downproject(feat)
        feat = feat.transpose(1, 2)
        return feat

    def encoder_input_ego(self, fmat):
        assert self.num_patches_ego == 1
        return self._encoder_input_ego(fmat)

    def _encoder_input_ego(self, fmat):
        B, T, _ = fmat.shape
        ego_feat = self.fmat_to_feature(fmat)
        ego_feat = rearrange(
            ego_feat, "(b n) t m -> b t n m", b=B, n=1, t=T, m=self.embed_dim_ego
        )
        return ego_feat

    def _preprocess_bbox_feat(self, feat, bbox_feat):
        B, _, T, c = feat.shape
        if c != self.embed_dim_hoi:
            feat = self.downproject(feat)
        bbox_feat = bbox_feat.view(-1, 4)
        bbox_feat = self.bbox_to_feature(bbox_feat)
        bbox_feat = bbox_feat.view(B, -1, T, self.coord_dim)
        return feat, bbox_feat

    def _fuse(self, ho_feat, bbox_feat):
        B, _, T, _ = ho_feat.shape
        feat = torch.cat((ho_feat, bbox_feat), dim=-1)
        feat = feat.view(-1, self.embed_dim_hoi + self.coord_dim)
        feat = self.feat_fusion(feat)
        feat = feat.view(B, -1, T, self.embed_dim_hoi)
        feat = feat.transpose(1, 2)
        return feat

    # original encoder
    def _encoder_input(self, feat, bbox_feat, src_mask):
        feat, bbox_feat = self._preprocess_bbox_feat(feat, bbox_feat)
        if self.use_flow:
            ho_feat_rgb, ho_feat_flow = feat[:, :4, :, :], feat[:, 4:, :, :]
            bbox_feat_rgb, bbox_feat_flow = bbox_feat[:, :4, :, :], bbox_feat[:, 4:, :, :]
            feat_rgb = self._fuse(ho_feat_rgb, bbox_feat_rgb)
            feat_flow = self._fuse(ho_feat_flow, bbox_feat_flow)
            feat = torch.cat((feat_rgb, feat_flow), dim=2)
        else:
            ho_feat = feat
            feat = self._fuse(ho_feat, bbox_feat)
        return feat, src_mask

    def indicator_function(self, pred_with_gt, pred_without_gt, mask):
        mask = mask[:, None]
        pred = pred_with_gt * mask + pred_without_gt * (1 - mask)
        return pred

    def auto_regressive_condition(
            self, last_obs_hand, target_hands, memory_hand, memory_mask_hand, hands_mask,
            last_obs_ego, target_egos, memory_ego, memory_mask_ego, egos_mask
    ):
        T = target_hands.shape[2]
        # [B, 1, 512]
        future_hand = rearrange(
            last_obs_hand, "(b l) n m -> b l (n m)", b=self.B, l=1, n=2, m=2
        )
        # condition_hand = self.learnable_token_left.expand(self.B, 1, 512)
        condition_hand = self.learnable_token_hand.expand(self.B, 1, 512)
        target_hands = rearrange(
            target_hands, "b n t m -> (b n) t m", b=self.B, n=2, t=T, m=2
        )
        hands_mask = rearrange(hands_mask, "b n t -> (b n) t", b=self.B, n=2, t=T)
        future_ego = last_obs_ego.unsqueeze(1)
        condition_ego = self.learnable_token_ego.expand(self.B, 1, self.embed_dim_ego)

        # time series loop
        for t in range(T):
            # mask for ground truth but does't mask anything in fact
            trg_mask_hand = torch.ones_like(condition_hand[:, 0:1, 0])
            trg_mask_hand = get_subsequent_mask(trg_mask_hand)
            trg_mask_ego = torch.ones_like(condition_ego[:, 0:1, 0])
            trg_mask_ego = get_subsequent_mask(trg_mask_ego)

            # x_hand: [B, token, 512]
            # x_ego: [B, token, 512]
            x_hand = self.decoder_hoi(
                condition_hand, memory_hand, memory_mask=memory_mask_hand, trg_mask=trg_mask_hand
            )
            x_ego = self.decoder_ego(
                condition_ego, memory_ego, memory_mask=memory_mask_ego, trg_mask=trg_mask_ego
            )
            # retrieve context of the last token: [B, 512] and [B, 512]
            x_hand = x_hand[:, -1, :]
            # concat tokens
            condition_hand = torch.cat((condition_hand, x_hand.unsqueeze(1)), dim=1)

            x_ego = x_ego[:, -1, :]
            condition_ego = torch.cat((condition_ego, x_ego.unsqueeze(1)), dim=1)

            pred_hand = self.hand_head(x_hand)
            pred_ego = self.ego_head(x_ego)
            # concat predicted hand with previous tokens
            future_hand = torch.cat((future_hand, pred_hand.unsqueeze(1)), dim=1)
            future_ego = torch.cat((future_ego, pred_ego.unsqueeze(1)), dim=1)

        # rearrange
        future_hand = rearrange(
            future_hand[:, 1:, :], "b t (n m) -> b n t m", b=self.B, n=2, t=T, m=2
        )
        future_ego = rearrange(
            future_ego[:, 1:, :], "(b n) t m -> b n t m", b=self.B, n=1, t=T, m=9
        )

        return future_hand, future_ego

    def forward(self, feat, bbox_feat, ego_input, attention_mask, future_hands, future_egos, hands_mask, egos_mask):
        # feat: (B, 5, To, src_in_features), global, obj, hand
        # (feat with flow: (B, 10, To, src_in_features))
        # bbox_feat: (B, 4, To, 4), obj & hand
        # (bbox_feat with flow: (B, 8, To, 4))
        # fmats_input: (B, To_ego, 9)
        # attention_mask: (B, To, 4), obj & hand
        # future_hands: (B, 2, Tp, 2) left & right, T=5 (does not contain last observation frame)
        # futuer_egos: (B, Tp_ego, 9) T=5
        # hands_mask: (B, 2, Tp), left & right traj valid
        # egos_mask: (B, Tp_ego)
        # ref_pos: (B), the index of reference frame for relativization
        # return: pred_hands, pred_egos, traj_loss, traj_kl_loss, motion_loss, motion_kl_loss

        self.B = bbox_feat.shape[0]
        # if necessary, add global_mask (all one tensor) to the valid_mask: [B, T, 4] -> [B, T, 10]
        assert attention_mask.shape[2] == 4
        src_mask_global = torch.ones_like(
            attention_mask[:, :, 0:1],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        src_mask_global = torch.cat((src_mask_global, src_mask_global), dim=2)
        src_mask_hoi = attention_mask
        src_mask_ego = torch.ones_like(
            ego_input[:, :, 0:1],
            dtype=src_mask_global.dtype,
            device=src_mask_global.device
        )
        src_mask = torch.cat(([src_mask_global, src_mask_hoi, src_mask_ego]), dim=2)
        global_feat = torch.cat((feat[:, 0, :, :].unsqueeze(1), feat[:, 5, :, :].unsqueeze(1)), dim=1)

        # fuse feat and bbox_feat before going through transformer encoder
        # hoi_feat, src_mask_hoi = self.encoder_input_hoi(hoi_feat, bbox_feat, src_mask_hoi)
        hoi_feat = self.encoder_input_hoi_only(bbox_feat)
        global_feat = self.encoder_input_global(global_feat)
        ego_feat = self.encoder_input_ego(ego_input)

        feat = torch.cat(([global_feat, hoi_feat, ego_feat]), dim=2)
        x = self.encoder_hoi(feat, mask=src_mask)

        memory_hand = x[:, -1, 4:6, :]
        memory_ego = x[:, -1, 6:7, :]

        # [B, 1, 1] for memory mask single hand
        memory_mask_hand = get_pad_mask(src_mask[:, -1, 4:6], pad_idx=0)
        memory_mask_ego = get_pad_mask(src_mask[:, -1, 6:7], pad_idx=0)

        # retrieve only hand bboxes of the last observable frame
        # in the order of left and right hand
        observe_bbox = bbox_feat[:, 2:4, -1, :]
        # calculate the center coordinate of the bbox for left and right hand
        # [B, 2, 4] -> [B, 2, 2]
        last_obs_hand = (observe_bbox[:, :, :2] + observe_bbox[:, :, 2:]) / 2

        # retrieve ego vectors of the last observable frame
        # [B, T, 2] -> [B, 2]
        last_obs_ego = ego_input[:, -1, :]

        # auto-regressive prediction loop
        pred_hands, pred_egos = self.auto_regressive_condition(
            last_obs_hand, future_hands, memory_hand, memory_mask_hand, hands_mask,
            last_obs_ego, future_egos, memory_ego, memory_mask_ego, egos_mask,
        )
        return pred_hands, pred_egos
