"""The existing Notebook 08 dual-stream model, reusable outside notebooks.

Preserves its parameter names and architecture so historical checkpoints load.
This extraction does not establish that dual pooling improves performance.
"""

import timm
import torch
from torch import nn


class DualStreamSeverityViT(nn.Module):
    def __init__(self, num_classes=3, drop_path_rate=0.1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        embed_dim = self.backbone.embed_dim
        self.norm = self.backbone.norm
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        combined = torch.cat([features[:, 0], features[:, 1:].mean(dim=1)], dim=-1)
        return self.head(combined)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, num_blocks=4):
        for block in self.backbone.blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
