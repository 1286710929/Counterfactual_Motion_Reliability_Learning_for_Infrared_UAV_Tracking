"""
Basic OSTrack model adapted to this FocusTrack codebase.
"""
import os
import math
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.focustrack.vit import vit_base_patch16_224
from lib.models.layers.box_head import build_box_head
from lib.models.ostrack.motion_encoder import build_motion_encoder
from lib.utils.box_ops import box_xyxy_to_cxcywh


class MotionPromptPool(nn.Module):
    """Pool dense motion tokens into a small set of Transformer prompt tokens."""

    def __init__(self, in_dim, out_dim, num_prompts):
        super().__init__()
        self.num_prompts = num_prompts
        self.query = nn.Parameter(torch.empty(1, num_prompts, out_dim))
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, motion_tokens):
        batch_size = motion_tokens.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        key = self.key(motion_tokens)
        value = self.value(motion_tokens)
        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.shape[-1])
        prompt = torch.matmul(attn.softmax(dim=-1), value)
        return self.norm(prompt)


class MotionScoreFusion(nn.Module):
    """Learn a residual motion-aware correction for the localization response."""

    def __init__(self, hidden_dim=16, alpha=1.0, residual_scale=1.0, use_reliability=False):
        super().__init__()
        self.alpha = float(alpha)
        self.residual_scale = float(residual_scale)
        self.use_reliability = bool(use_reliability)
        self.last_reliability = None
        self.fusion = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        if self.use_reliability:
            self.reliability_head = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.reliability_head = None

    def forward(self, score_map, motion_map):
        eps = 1e-6
        score_logit = torch.logit(score_map.clamp(eps, 1.0 - eps))
        residual = self.fusion(torch.cat([score_map, motion_map], dim=1))
        residual = torch.tanh(residual) * self.residual_scale
        fused_score = torch.sigmoid(score_logit + residual)
        reliability = self._estimate_reliability(score_map, motion_map)
        self.last_reliability = reliability.detach()
        return score_map + self.alpha * reliability[:, :, None, None] * (fused_score - score_map)

    def _estimate_reliability(self, score_map, motion_map):
        if self.reliability_head is None:
            return score_map.new_ones(score_map.shape[0], 1)

        score_peak = score_map.flatten(1).amax(dim=1, keepdim=True)
        motion_peak = motion_map.flatten(1).amax(dim=1, keepdim=True)
        score_center = self._soft_argmax_2d(score_map)
        motion_center = self._soft_argmax_2d(motion_map)
        peak_dist = (score_center - motion_center).norm(dim=1, keepdim=True) / math.sqrt(2.0)
        features = torch.cat([score_peak, motion_peak, peak_dist.clamp(0.0, 1.0)], dim=1)
        return self.reliability_head(features)

    @staticmethod
    def _soft_argmax_2d(score_map, temperature=0.1):
        b, _, h, w = score_map.shape
        prob = torch.softmax(score_map.flatten(1) / temperature, dim=1)
        ys = (torch.arange(h, device=score_map.device, dtype=score_map.dtype) + 0.5) / h
        xs = (torch.arange(w, device=score_map.device, dtype=score_map.dtype) + 0.5) / w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        return prob @ coords


class OSTrack(nn.Module):
    """One-stream tracking model with a ViT backbone and a box head."""

    def __init__(self, transformer, box_head, motion_encoder=None, cfg=None, hidden_dim=None,
                 aux_loss=False, box_head_type="CENTER"):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.motion_encoder = motion_encoder
        self.cfg = cfg
        self.aux_loss = aux_loss
        self.box_head_type = box_head_type
        self.use_motion = motion_encoder is not None
        self.use_token_gate = bool(getattr(cfg.MODEL.MOTION, "TOKEN_GATE", False)) if cfg is not None else False
        self.token_gate_alpha = float(getattr(cfg.MODEL.MOTION, "TOKEN_GATE_ALPHA", 0.5)) if cfg is not None else 0.5
        self.use_score_fusion = bool(getattr(cfg.MODEL.MOTION, "SCORE_FUSION", False)) if cfg is not None else False
        self.motion_score_fusion = None
        if self.use_score_fusion:
            self.motion_score_fusion = MotionScoreFusion(
                alpha=getattr(cfg.MODEL.MOTION, "SCORE_FUSION_ALPHA", 1.0),
                residual_scale=getattr(cfg.MODEL.MOTION, "SCORE_FUSION_RESIDUAL_SCALE", 1.0),
                use_reliability=getattr(cfg.MODEL.MOTION, "SCORE_FUSION_RELIABILITY", False),
            )
        self.use_motion_prompt = bool(getattr(cfg.MODEL.MOTION, "PROMPT_ENABLE", False)) if cfg is not None else False
        self.motion_prompt_pool = None
        if self.use_motion_prompt:
            if motion_encoder is None:
                raise ValueError("MODEL.MOTION.ENABLE is required when MODEL.MOTION.PROMPT_ENABLE is True.")
            if hidden_dim is None:
                raise ValueError("hidden_dim is required when MODEL.MOTION.PROMPT_ENABLE is True.")
            motion_dim = int(getattr(cfg.MODEL.MOTION, "HIDDEN_DIM", hidden_dim))
            num_prompts = int(getattr(cfg.MODEL.MOTION, "PROMPT_NUM", 4))
            self.motion_prompt_pool = MotionPromptPool(motion_dim, hidden_dim, num_prompts)
        self.use_exist = bool(getattr(cfg.MODEL.EXIST, "ENABLE", False)) if cfg is not None else False
        self.exist_head = None
        if self.use_exist:
            if hidden_dim is None:
                raise ValueError("hidden_dim is required when MODEL.EXIST.ENABLE is True.")
            self.exist_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )

        if box_head_type in ["CORNER", "CENTER"]:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        else:
            raise NotImplementedError

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor, search: torch.Tensor, search_history=None, training=True):
        motion_map = None
        motion_prompts = None
        if self.use_motion:
            motion_map = self.motion_encoder(search, search_history)
            motion_tokens = getattr(self.motion_encoder, "last_motion_tokens", None)
            if self.motion_prompt_pool is not None and motion_tokens is not None:
                motion_prompts = self.motion_prompt_pool(motion_tokens)

        x, aux_dict = self.backbone(z=template, x=search, prompt_tokens=motion_prompts)

        feat_last = x[-1] if isinstance(x, list) else x
        if motion_map is not None and self.use_token_gate:
            feat_last = self.apply_motion_token_gate(feat_last, motion_map)

        out = self.forward_box_head(cat_feature=feat_last)
        if motion_map is not None and self.motion_score_fusion is not None and "score_map" in out:
            out["motion_score_map"] = self.motion_score_fusion(out["score_map"], motion_map)
            out["motion_reliability"] = self.motion_score_fusion.last_reliability
        if self.exist_head is not None:
            search_feat = feat_last[:, -self.feat_len_s:]
            exist_logits = self.exist_head(search_feat.mean(dim=1)).squeeze(-1)
            out["exist_logits"] = exist_logits
            out["exist_prob"] = exist_logits.sigmoid()
        if motion_map is not None:
            out["motion_map"] = motion_map
        out.update(aux_dict)
        return out

    def apply_motion_token_gate(self, feat, motion_map):
        search_feat = feat[:, -self.feat_len_s:]
        motion_gate = motion_map.flatten(2).transpose(1, 2)
        gated_search_feat = search_feat * (1.0 + self.token_gate_alpha * motion_gate)
        return torch.cat([feat[:, :-self.feat_len_s], gated_search_feat], dim=1)

    def forward_box_head(self, cat_feature, gt_score_map=None):
        enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = enc_opt.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()
        bs, num_queries, channels, hw = opt.size()
        opt_feat = opt.view(-1, channels, self.feat_sz_s, self.feat_sz_s)

        if self.box_head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, None, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            return {
                "pred_boxes": outputs_coord.view(bs, num_queries, 4),
                "score_map": score_map,
            }

        if self.box_head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, None, gt_score_map)
            return {
                "pred_boxes": bbox.view(bs, num_queries, 4),
                "score_map": score_map_ctr,
                "size_map": size_map,
                "offset_map": offset_map,
            }

        raise NotImplementedError


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_ostrack(cfg, settings=None, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "../../../pretrained_models")
    pretrained = ""
    if cfg.MODEL.PRETRAIN_FILE and ("OSTrack" not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)

    if cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224":
        backbone = vit_base_patch16_224(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
            use_cls_token=cfg.MODEL.HEAD.CLS_HEAD.USE_CLS_TOKEN,
            num_classes=cfg.MODEL.HEAD.CLS_HEAD.NUM_CLASSES,
            out_indices=cfg.MODEL.BACKBONE.OUT_INDICES,
        )
        hidden_dim = backbone.embed_dim
        backbone.finetune_track(cfg=cfg, patch_start_index=1)
    else:
        raise NotImplementedError

    box_head = build_box_head(cfg, hidden_dim, add_decoder=False)
    motion_encoder = None
    if getattr(cfg.MODEL.MOTION, "ENABLE", False):
        motion_encoder = build_motion_encoder(
            version=getattr(cfg.MODEL.MOTION, "VERSION", "v1"),
            search_size=cfg.DATA.SEARCH.SIZE,
            stride=cfg.MODEL.BACKBONE.STRIDE,
            hidden_dim=cfg.MODEL.MOTION.HIDDEN_DIM,
            history_len=cfg.MODEL.MOTION.HISTORY_LEN,
            num_heads=getattr(cfg.MODEL.MOTION, "NUM_HEADS", 4),
            num_layers=getattr(cfg.MODEL.MOTION, "NUM_LAYERS", 1),
            ring_inner=getattr(cfg.MODEL.MOTION, "RING_INNER", 0),
            ring_outer=getattr(cfg.MODEL.MOTION, "RING_OUTER", 2),
            dropout=getattr(cfg.MODEL.MOTION, "DROPOUT", 0.0),
        )

    model = OSTrack(
        backbone,
        box_head,
        motion_encoder=motion_encoder,
        cfg=cfg,
        hidden_dim=hidden_dim,
        aux_loss=False,
        box_head_type=cfg.MODEL.HEAD.BOX_HEAD.TYPE,
    )

    if cfg.MODEL.PRETRAIN_FILE and ("OSTrack" in cfg.MODEL.PRETRAIN_FILE) and training:
        checkpoint = _load_checkpoint(cfg.MODEL.PRETRAIN_FILE)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print("Load pretrained OSTrack model from: " + cfg.MODEL.PRETRAIN_FILE)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)

    return model
