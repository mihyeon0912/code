# model.py  (코드명: MODEL_RGB_CONTRAST_CONCAT_V1)
# 역할:
#  - forward(rgb_norm, contrast_bw):
#      * rgb_norm    : (B,3,224,224) ImageNet 정규화 RGB
#      * contrast_bw : (B,3,224,224) [0,1], 배경은 (0,1,0) green, 전경은 흑/백(R=G=B)
#  - contrast_bw에서 green 배경을 마스크로 제거하고 1채널 패턴 추출
#  - ImageBackbone(ResNet50, RGB) + PatternBackbone(ResNet18, 1ch 패턴)
#  - 두 feature map을 채널 concat → GAP → Linear → L2 normalize

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =========================================================
# Backbones
# =========================================================
class ImageBackbone(nn.Module):
    """
    ResNet50 백본에서 feature map 추출 후 1x1 conv로 d 채널 정렬.
    입력: (B,3,224,224)  [ImageNet 정규화]
    출력: (B, d, h, w)  (보통 h=w=7)
    """
    def __init__(self, d: int = 256, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.proj = nn.Conv2d(2048, d, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # (B,2048,h,w)
        x = self.proj(x)            # (B,d,h,w)
        x = self.bn(x)
        return x


class PatternBackbone(nn.Module):
    """
    ResNet18 기반 1채널 입력 백본.
    contrast_bw에서 추출한 1채널 패턴을 입력으로 사용.
    입력: (B,1,224,224) in [0,1]
    출력: (B, d, h, w)  (보통 h=w=7)
    """
    def __init__(self, d: int = 256, pretrained: bool = True):
        super().__init__()
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # conv1을 1ch용으로 교체 (pretrained conv 평균 사용)
        old_conv: nn.Conv2d = base.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained and old_conv.weight.shape[1] == 3:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        base.conv1 = new_conv

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.proj = nn.Conv2d(512, d, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # (B,512,h,w)
        x = self.proj(x)            # (B,d,h,w)
        x = self.bn(x)
        return x


# =========================================================
# contrast_bw → 1채널 패턴
# =========================================================
@torch.no_grad()
def extract_pattern_from_contrast_bw(
    contrast_rgb_01: torch.Tensor,
    green_g_thresh: float = 0.7,
    green_rb_thresh: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    contrast_rgb_01: (B,3,H,W) in [0,1]
      - 배경: (0,1,0) 근처 green
      - 전경: R=G=B (0 or 1) 흑/백 패턴

    returns:
      pattern_1ch: (B,1,H,W) in [0,1], 배경은 0
      keep_hw    : (B,H,W) bool, 전경 True
    """
    R = contrast_rgb_01[:, 0]
    G = contrast_rgb_01[:, 1]
    B = contrast_rgb_01[:, 2]

    # green 배경: G 크고, R/B 작고, G 자체도 충분히 큼
    green_bg = (G > green_g_thresh) & (R < green_rb_thresh) & (B < green_rb_thresh)
    keep_hw = ~green_bg  # 전경 True

    # 전경 영역에서 R,G,B가 거의 같으므로 채널 평균 사용
    gray = contrast_rgb_01.mean(dim=1, keepdim=True)   # (B,1,H,W)
    pattern = gray * keep_hw.unsqueeze(1).float()

    return pattern, keep_hw


# =========================================================
# CombinedEmbedding (RGB + contrast_bw concat)
# =========================================================
class CombinedEmbedding(nn.Module):
    """
    최종 임베딩 = ImageBackbone(RGB) + PatternBackbone(contrast_bw→1ch pattern)

    구조:
      1) rgb_norm      → ImageBackbone(ResNet50)       → img_fm (B,d,h,w)
      2) contrast_bw   → extract_pattern_from_contrast_bw → pat_1ch
                       → PatternBackbone(ResNet18,1ch) → pat_fm (B,d,h,w)
      3) concat(ch)    → feat (B,2d,h,w)
      4) GAP           → (B,2d)
      5) Linear(2d→fused_dim) → L2 normalize
    """
    def __init__(
        self,
        d: int = 256,
        fused_dim: int = 128,
        nblocks: int = 0,          # (호환용, 사용하지 않음)
        nheads: int = 0,           # (호환용, 사용하지 않음)
        pdrop: float = 0.0,        # (호환용, 사용하지 않음)
        mlp_ratio: float = 4.0,    # (호환용, 사용하지 않음)
        pretrained_backbones: bool = True,
        # 아래 두 파라미터는 이전 버전과 호환을 위해 남겨두지만 사용하지 않음
        keep_thresh: float = 0.05,
        min_fg_ratio: float = 0.01,
    ):
        super().__init__()
        self.d = d
        self.img_backbone = ImageBackbone(d=d, pretrained=pretrained_backbones)
        self.pat_backbone = PatternBackbone(d=d, pretrained=pretrained_backbones)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(2 * d, fused_dim)
        self.embed_dim = fused_dim

    def forward(self, rgb: torch.Tensor, contrast_rgb_01: torch.Tensor) -> torch.Tensor:
        """
        rgb            : (B,3,224,224), ImageNet 정규화
        contrast_rgb_01: (B,3,224,224), [0,1], 배경 green + 전경 흑/백
        """
        # 1) RGB branch
        img_fm = self.img_backbone(rgb)  # (B,d,h,w)

        # 2) contrast_bw → 1ch pattern → pattern branch
        pat_1ch, _ = extract_pattern_from_contrast_bw(contrast_rgb_01)  # (B,1,H,W)
        pat_fm = self.pat_backbone(pat_1ch)                             # (B,d,h,w)

        # 3) concat + GAP
        feat = torch.cat([img_fm, pat_fm], dim=1)  # (B,2d,h,w)
        pooled = self.gap(feat).flatten(1)         # (B,2d)

        # 4) Linear + L2 normalize
        z = self.head(pooled)                      # (B,fused_dim)
        z = F.normalize(z, dim=1, eps=1e-6)
        return z