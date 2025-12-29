# model.py
# CODE NAME: MODEL_CROSS_ATTENTION_PATTERN_V_FINAL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
import math


# =========================================================
# position_embedding
# =========================================================
def build_2d_sincos_position_embedding(h, w, dim, device):
    """
    2D sine-cosine positional embedding
    Return: (1, H*W, dim)
    """
    assert dim % 4 == 0, "dim must be divisible by 4"

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    omega = torch.arange(dim // 4, device=device) / (dim // 4)
    omega = 1.0 / (10000 ** omega)

    out_y = torch.einsum("hw,d->hwd", grid_y, omega)
    out_x = torch.einsum("hw,d->hwd", grid_x, omega)

    pe = torch.cat(
        [
            torch.sin(out_x),
            torch.cos(out_x),
            torch.sin(out_y),
            torch.cos(out_y),
        ],
        dim=-1
    )

    pe = pe.view(1, h * w, dim)
    return pe

# =========================================================
# Backbones
# =========================================================
class ImageBackbone(nn.Module):
    def __init__(self, d=256, pretrained=True):
        super().__init__()
        net = resnet50(pretrained=pretrained)
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.proj = nn.Conv2d(2048, d, 1)

    def forward(self, x):
        return self.proj(self.body(x))


class PatternBackbone(nn.Module):
    def __init__(self, d=256, pretrained=True):
        super().__init__()
        net = resnet18(pretrained=pretrained)
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.proj = nn.Conv2d(512, d, 1)

    def forward(self, x):
        return self.proj(self.body(x))


# =========================================================
# SE on pattern
# =========================================================
class PatternSE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d, d // 4, 1),
            nn.ReLU(),
            nn.Conv2d(d // 4, d, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * (1.0 + self.fc(x))



# =========================================================
# Cross Attention
# =========================================================
class CrossAttention(nn.Module):
    def __init__(self, d, nheads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=nheads,
            batch_first=True,
        )

    def forward(self, q, kv, key_padding_mask):
        out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=~key_padding_mask
        )
        return out


# =========================================================
# Combined Model
# =========================================================
class CombinedEmbedding(nn.Module):
    def __init__(self, d=256, fused_dim=128, nheads=4):
        super().__init__()
        self.img_backbone = ImageBackbone(d)
        self.pat_backbone = PatternBackbone(d)
        self.pattern_se = PatternSE(d)
        self.cross_attn = CrossAttention(d, nheads)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d, fused_dim)

    def forward(self, rgb, pattern_1ch, seg_mask_hw):
        img_fm = self.img_backbone(rgb)      # (B,d,H,W)
        pat_fm = self.pattern_se(
            self.pat_backbone(pattern_1ch)
        )

        B, C, H, W = img_fm.shape
        img_tok = img_fm.flatten(2).transpose(1, 2)   # (B,N,d)
        pat_tok = pat_fm.flatten(2).transpose(1, 2)

        # -------------------------------
        # 2D Positional Encoding
        # -------------------------------
        pos = build_2d_sincos_position_embedding(
            h=H,
            w=W,
            dim=C,
            device=img_tok.device
        )  # (1, N, C)

        img_tok = img_tok + pos
        pat_tok = pat_tok + pos

        mask = F.interpolate(
            seg_mask_hw.unsqueeze(1).float(),
            size=(H, W),
            mode="nearest"
        ).flatten(2).squeeze(1).bool()

        attn = self.cross_attn(img_tok, pat_tok, mask)
        pooled = self.gap(attn.transpose(1, 2)).squeeze(-1)

        z = F.normalize(self.head(pooled), dim=1)
        return z