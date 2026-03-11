import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Keep only valid class ids [0, num_classes-1], excluding ignore_index.
        mask = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < num_classes)
        )
        targets_valid = targets.clone()
        targets_valid[~mask] = 0

        targets_one_hot = F.one_hot(targets_valid, num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Apply mask
        mask = mask.unsqueeze(1).float()
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask

        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes=19,
        dice_weight=0.5,
        ce_weight=0.5,
        ignore_index=255,
        ce_class_weights=None,
    ):
        super().__init__()
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
