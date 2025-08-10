from __future__ import annotations
import torch
import torch.nn as nn


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - Dice. logits: [B,1,H,W], target: [B,1,H,W] in {0,1}."""
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = (prob + target).sum(dim=(1, 2, 3)).clamp_min(eps)
    return (1 - (num + eps) / (den + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = float(bce_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w * self.bce(logits, target) + (1 - self.w) * dice_loss(logits, target)
