# Triplet Loss 구현
# Anchor-P는 가깝게, Anchor-N는 멀게 학습시키는 손실 함수
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 임베딩은 이미 L2 정규화됨 (모델에서)
        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)
        losses = torch.clamp(d_ap - d_an + self.margin, min=0.0)
        return losses.mean()