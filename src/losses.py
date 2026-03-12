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


class BoundaryWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, class_weights=None, boundary_factor=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.boundary_factor = boundary_factor

    def _compute_boundary_mask(self, targets):
        valid = targets != self.ignore_index
        padded = F.pad(targets, (1, 1, 1, 1), mode='replicate')

        center = padded[:, 1:-1, 1:-1]
        up = padded[:, :-2, 1:-1]
        down = padded[:, 2:, 1:-1]
        left = padded[:, 1:-1, :-2]
        right = padded[:, 1:-1, 2:]

        boundary = (
            ((center != up) & valid) |
            ((center != down) & valid) |
            ((center != left) & valid) |
            ((center != right) & valid)
        )
        return boundary.float()

    def forward(self, logits, targets):
        ce_map = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        valid = (targets != self.ignore_index).float()
        if self.boundary_factor > 0:
            boundary = self._compute_boundary_mask(targets)
            pixel_weights = 1.0 + self.boundary_factor * boundary
        else:
            pixel_weights = torch.ones_like(ce_map)

        weighted = ce_map * pixel_weights * valid
        normalizer = (pixel_weights * valid).sum().clamp_min(1.0)
        return weighted.sum() / normalizer


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, class_weights=None, gamma=2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_map = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        valid = (targets != self.ignore_index).float()
        pt = torch.exp(-ce_map)
        focal_factor = (1.0 - pt).pow(self.gamma)
        weighted = focal_factor * ce_map * valid
        normalizer = valid.sum().clamp_min(1.0)
        return weighted.sum() / normalizer


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes=19,
        dice_weight=0.5,
        ce_weight=0.5,
        ignore_index=255,
        ce_class_weights=None,
        boundary_ce_factor=0.0,
        ce_type='ce',
        focal_gamma=2.0,
    ):
        super().__init__()
        self.dice = DiceLoss(ignore_index=ignore_index)
        if ce_type == 'ce':
            self.ce = BoundaryWeightedCrossEntropyLoss(
                ignore_index=ignore_index,
                class_weights=ce_class_weights,
                boundary_factor=boundary_ce_factor,
            )
        elif ce_type == 'focal':
            self.ce = FocalCrossEntropyLoss(
                ignore_index=ignore_index,
                class_weights=ce_class_weights,
                gamma=focal_gamma,
            )
        else:
            raise ValueError(f"Unknown ce_type: {ce_type}")
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
