# ResNet50을 백본으로 임베딩 벡터를 생성하는 모델을 정의
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet50Embedding(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        # torchvision 버전별 호환
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        except Exception:
            self.backbone = models.resnet50(pretrained=pretrained)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_dim, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)  # L2 정규화
        return x
        