import torch
from torch import nn
import torch.nn.functional as F
import math


def _build_2d_sincos_pos_embed(height, width, dim, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pos = torch.stack((x, y), dim=-1).reshape(-1, 2)

    half_dim = dim // 2
    freq_dim = max(1, half_dim // 2)
    omega = torch.arange(freq_dim, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / freq_dim))

    out_x = pos[:, 0:1] * omega[None, :]
    out_y = pos[:, 1:2] * omega[None, :]
    emb = torch.cat((out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()), dim=1)

    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    elif emb.shape[1] > dim:
        emb = emb[:, :dim]
    return emb.unsqueeze(0)


def _prepare_history(search, search_history, history_len, feat_size, neutral_value):
    if search_history is None:
        return None, search.new_full((search.shape[0], 1, feat_size, feat_size), neutral_value)

    if search_history.dim() != 5:
        raise ValueError("search_history should be a 5D tensor.")

    # Accept both [K, B, C, H, W] from the training loader and
    # [B, K, C, H, W] from the tracker.
    if search_history.shape[0] == search.shape[0]:
        history = search_history
    elif search_history.shape[1] == search.shape[0]:
        history = search_history.permute(1, 0, 2, 3, 4).contiguous()
    else:
        raise ValueError("search_history batch dimension does not match search.")

    if history.shape[1] == 0:
        return None, search.new_full((search.shape[0], 1, feat_size, feat_size), neutral_value)

    history = history[:, -history_len:]
    if history.shape[1] < history_len:
        pad = history[:, :1].repeat(1, history_len - history.shape[1], 1, 1, 1)
        history = torch.cat([pad, history], dim=1)
    return history, None


def _avg_pool_same(x, kernel_size):
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


class MotionEvidenceEncoderV1(nn.Module):
    """Lightweight raw temporal-change encoder kept for ablations.

    The module consumes the current search crop and one or more previous search
    crops, computes temporal high-pass evidence, and predicts a low-resolution
    confidence map aligned with the search feature map.
    """

    def __init__(self, search_size, stride, hidden_dim=16, history_len=1, neutral_value=0.0):
        super().__init__()
        self.search_size = search_size
        self.stride = stride
        self.feat_size = search_size // stride
        self.history_len = history_len
        self.neutral_value = neutral_value
        self.debug_maps = {}
        self.last_motion_tokens = None

        self.encoder = nn.Sequential(
            nn.Conv2d(history_len, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, search, search_history=None):
        history, neutral = _prepare_history(
            search, search_history, self.history_len, self.feat_size, self.neutral_value
        )
        if neutral is not None:
            self.debug_maps = {}
            self.last_motion_tokens = None
            return neutral

        diff = (search[:, None] - history).abs().mean(dim=2)
        motion_feat = self.encoder[:-1](diff)
        prompt_feat = F.interpolate(
            motion_feat,
            size=(self.feat_size, self.feat_size),
            mode="bilinear",
            align_corners=False,
        )
        self.last_motion_tokens = prompt_feat.flatten(2).transpose(1, 2)

        motion_logits = self.encoder[-1](motion_feat)
        motion_logits = F.interpolate(
            motion_logits,
            size=(self.feat_size, self.feat_size),
            mode="bilinear",
            align_corners=False,
        )
        motion_map = motion_logits.sigmoid()
        self.debug_maps = {"raw_diff": diff.mean(dim=1, keepdim=True).detach()}
        return motion_map


class MotionEvidenceEncoderV2(nn.Module):
    """Sparse tiny-target-biased temporal evidence encoder.

    V2 keeps the useful raw temporal cue from V1, but suppresses broad
    background-dominated temporal changes through local contrast filtering and
    emphasizes small isolated peaks before learnable refinement.
    """

    def __init__(self, search_size, stride, hidden_dim=16, history_len=1, neutral_value=0.0):
        super().__init__()
        self.search_size = search_size
        self.stride = stride
        self.feat_size = search_size // stride
        self.history_len = history_len
        self.neutral_value = neutral_value
        self.debug_maps = {}

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, search, search_history=None):
        history, neutral = _prepare_history(
            search, search_history, self.history_len, self.feat_size, self.neutral_value
        )
        if neutral is not None:
            self.debug_maps = {}
            return neutral

        diff_seq = (search[:, None] - history).abs().mean(dim=2)
        raw_diff = diff_seq.mean(dim=1, keepdim=True)

        residuals = []
        for kernel_size in (7, 15, 31):
            background = _avg_pool_same(raw_diff, kernel_size)
            residuals.append(F.relu(raw_diff - background))
        residual = torch.stack(residuals, dim=0).mean(dim=0)

        peak = F.max_pool2d(residual, kernel_size=3, stride=1, padding=1)
        context = _avg_pool_same(residual, 15)
        tiny_motion = F.relu(peak - context)

        motion_input = torch.cat([raw_diff, residual, tiny_motion], dim=1)
        motion_logits = self.encoder(motion_input)
        motion_logits = F.interpolate(
            motion_logits,
            size=(self.feat_size, self.feat_size),
            mode="bilinear",
            align_corners=False,
        )
        motion_map = motion_logits.sigmoid()

        self.debug_maps = {
            "raw_diff": raw_diff.detach(),
            "residual": residual.detach(),
            "tiny_motion": tiny_motion.detach(),
        }
        return motion_map


class ResidualMotionTransformer(nn.Module):
    """Background-referenced residual motion transformer.

    RMT compares the temporal change of each search token with the temporal
    changes of its local background ring. It keeps motion that is inconsistent
    with the surrounding background and suppresses shared camera/background
    changes without explicit global motion estimation.
    """

    def __init__(self, search_size, stride, hidden_dim=64, history_len=1,
                 neutral_value=0.0, num_heads=4, num_layers=1):
        super().__init__()
        self.search_size = search_size
        self.stride = stride
        self.feat_size = search_size // stride
        self.feat_len = self.feat_size ** 2
        self.history_len = history_len
        self.neutral_value = neutral_value
        self.debug_maps = {}

        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=stride, stride=stride)
        self.delta_norm = nn.LayerNorm(hidden_dim)
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.feat_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, search, search_history=None):
        history, neutral = _prepare_history(
            search, search_history, self.history_len, self.feat_size, self.neutral_value
        )
        if neutral is not None:
            self.debug_maps = {}
            return neutral

        prev_search = history[:, -1]
        curr_tokens_2d = self.patch_embed(search)
        prev_tokens_2d = self.patch_embed(prev_search)
        delta_2d = curr_tokens_2d - prev_tokens_2d

        delta_tokens = delta_2d.flatten(2).transpose(1, 2)
        delta_tokens = self.delta_norm(delta_tokens)

        bg_tokens = self._local_background_motion(delta_2d, delta_tokens)
        residual_tokens = self.residual_norm(delta_tokens - bg_tokens)
        encoded_tokens = self.encoder(residual_tokens + self.pos_embed)
        motion_logits = self.score_head(encoded_tokens)
        motion_logits = motion_logits.transpose(1, 2).reshape(
            search.shape[0], 1, self.feat_size, self.feat_size
        )
        motion_map = motion_logits.sigmoid()

        self.debug_maps = {
            "rmt_delta": self._token_energy(delta_tokens),
            "rmt_bg": self._token_energy(bg_tokens),
            "rmt_residual": self._token_energy(residual_tokens),
        }
        return motion_map

    def _local_background_motion(self, delta_2d, delta_tokens):
        b, c, h, w = delta_2d.shape
        local_tokens = F.unfold(delta_2d, kernel_size=3, padding=1)
        local_tokens = local_tokens.view(b, c, 9, h * w).permute(0, 3, 2, 1).contiguous()
        ring_tokens = torch.cat([local_tokens[:, :, :4], local_tokens[:, :, 5:]], dim=2)
        ring_tokens = self.delta_norm(ring_tokens)

        query = delta_tokens.unsqueeze(2)
        attn = (query * ring_tokens).sum(dim=-1) / math.sqrt(c)
        attn = attn.softmax(dim=-1)
        bg_tokens = (attn.unsqueeze(-1) * ring_tokens).sum(dim=2)
        return bg_tokens

    def _token_energy(self, tokens):
        energy = tokens.pow(2).mean(dim=-1).sqrt()
        return energy.reshape(tokens.shape[0], 1, self.feat_size, self.feat_size).detach()


class LocalBackgroundReferenceAttention(nn.Module):
    def __init__(self, dim, num_heads=4, ring_inner=0, ring_outer=2, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("MODEL.MOTION.HIDDEN_DIM must be divisible by MODEL.MOTION.NUM_HEADS")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ring_outer = ring_outer
        self.kernel_size = ring_outer * 2 + 1

        offsets = []
        for dy in range(-ring_outer, ring_outer + 1):
            for dx in range(-ring_outer, ring_outer + 1):
                dist = max(abs(dx), abs(dy))
                offsets.append(dist > ring_inner and dist <= ring_outer)
        self.register_buffer("ring_mask", torch.tensor(offsets, dtype=torch.bool), persistent=False)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, delta_map, delta_tokens):
        b, c, h, w = delta_map.shape
        n = h * w
        k_area = self.kernel_size * self.kernel_size

        neighborhood = F.unfold(delta_map, kernel_size=self.kernel_size, padding=self.ring_outer)
        neighborhood = neighborhood.transpose(1, 2).reshape(b, n, c, k_area).permute(0, 1, 3, 2)

        valid = F.unfold(
            torch.ones((b, 1, h, w), device=delta_map.device, dtype=delta_map.dtype),
            kernel_size=self.kernel_size,
            padding=self.ring_outer,
        )
        valid = valid.transpose(1, 2).bool()
        bg_mask = valid & self.ring_mask.view(1, 1, k_area)

        q = self.q(delta_tokens).view(b, n, self.num_heads, self.head_dim)
        k = self.k(neighborhood).view(b, n, k_area, self.num_heads, self.head_dim)
        v = self.v(neighborhood).view(b, n, k_area, self.num_heads, self.head_dim)

        attn = torch.einsum("bnhd,bnkhd->bnhk", q, k) * self.scale
        attn = attn.masked_fill(~bg_mask[:, :, None, :], -1e4)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        bg_tokens = torch.einsum("bnhk,bnkhd->bnhd", attn, v).reshape(b, n, c)
        return self.proj(bg_tokens)


class ResidualMotionTransformerV4(nn.Module):
    """Full background-referenced residual motion transformer.

    V4 estimates a local background-reference motion token with multi-head
    attention over a configurable ring neighborhood, subtracts it from each
    candidate temporal-change token, and predicts a residual motion score map.
    """

    def __init__(self, search_size, stride, hidden_dim=128, history_len=1,
                 neutral_value=0.0, num_heads=4, num_layers=2,
                 ring_inner=0, ring_outer=2, dropout=0.0):
        super().__init__()
        self.search_size = search_size
        self.stride = stride
        self.feat_size = search_size // stride
        self.history_len = history_len
        self.neutral_value = neutral_value
        self.debug_maps = {}

        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=stride, stride=stride, bias=False)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.bg_ref = LocalBackgroundReferenceAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            ring_inner=ring_inner,
            ring_outer=ring_outer,
            dropout=dropout,
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, search, search_history=None):
        history, neutral = _prepare_history(
            search, search_history, self.history_len, self.feat_size, self.neutral_value
        )
        if neutral is not None:
            self.debug_maps = {}
            return neutral

        prev_search = history[:, -1]
        prev_feat = self.patch_embed(prev_search)
        curr_feat = self.patch_embed(search)

        b, c, h, w = curr_feat.shape
        prev_tokens = prev_feat.flatten(2).transpose(1, 2)
        curr_tokens = curr_feat.flatten(2).transpose(1, 2)

        delta_tokens = self.token_norm(curr_tokens - prev_tokens)
        delta_map = self._tokens_to_map(delta_tokens, h, w)

        bg_tokens = self.bg_ref(delta_map, delta_tokens)
        residual_tokens = self.residual_norm(delta_tokens - bg_tokens)

        pos = _build_2d_sincos_pos_embed(h, w, c, residual_tokens.device, residual_tokens.dtype)
        residual_tokens = self.encoder(residual_tokens + pos)
        self.last_motion_tokens = residual_tokens
        residual_map = self._tokens_to_map(residual_tokens, h, w)
        motion_map = torch.sigmoid(self.score_head(residual_map))

        self.debug_maps = {
            "rmt_delta": self._token_energy(delta_tokens, h, w),
            "rmt_bg": self._token_energy(bg_tokens, h, w),
            "rmt_residual": self._token_energy(residual_tokens, h, w),
        }
        return motion_map

    @staticmethod
    def _tokens_to_map(tokens, height, width):
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, height, width)

    @staticmethod
    def _token_energy(tokens, height, width):
        energy = tokens.pow(2).mean(dim=-1).sqrt()
        return energy.reshape(tokens.shape[0], 1, height, width).detach()


def build_motion_encoder(version, search_size, stride, hidden_dim=16, history_len=1,
                         neutral_value=0.0, num_heads=4, num_layers=1,
                         ring_inner=0, ring_outer=2, dropout=0.0):
    version = str(version).lower()
    if version == "v1":
        return MotionEvidenceEncoderV1(search_size, stride, hidden_dim, history_len, neutral_value)
    if version == "v2":
        return MotionEvidenceEncoderV2(search_size, stride, hidden_dim, history_len, neutral_value)
    if version == "v3":
        return ResidualMotionTransformer(
            search_size, stride, hidden_dim, history_len, neutral_value, num_heads, num_layers
        )
    if version == "v4":
        return ResidualMotionTransformerV4(
            search_size, stride, hidden_dim, history_len, neutral_value,
            num_heads, num_layers, ring_inner, ring_outer, dropout
        )
    raise ValueError("Unsupported motion encoder version: {}".format(version))


MotionEvidenceEncoder = MotionEvidenceEncoderV1
