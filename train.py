import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from dataset import FaceParsingDataset
from losses import CombinedLoss


def get_model(arch, num_classes, base_ch):
    if arch == 'attention_unet':
        from model import AttentionUNet, count_parameters
        model = AttentionUNet(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    elif arch == 'deeplab':
        from model_b_deeplab import LightDeepLab, count_parameters
        model = LightDeepLab(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    elif arch == 'segnet':
        from model_c_segnet import LightSegNet, count_parameters
        model = LightSegNet(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    total = sum(p.numel() for p in model.parameters())
    print(f"[{arch}] Parameters: {total:,} / 1,821,085 limit")
    assert total < 1821085, f"Parameter limit exceeded! ({total:,})"
    return model


def compute_f_measure(preds, targets, num_classes, ignore_index=255):
    f_scores = []
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum()
        fp = ((preds == cls) & (targets != cls)).sum()
        fn = ((preds != cls) & (targets == cls)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f = 2 * precision * recall / (precision + recall + 1e-8)
        f_scores.append(f)

    return np.mean(f_scores)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f_score = compute_f_measure(all_preds, all_targets, num_classes)

    return total_loss / len(loader), f_score


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Arch: {args.arch}")

    train_dataset = FaceParsingDataset(
        img_dir=os.path.join(args.data_dir, 'train', 'images'),
        mask_dir=os.path.join(args.data_dir, 'train', 'masks'),
        img_size=args.img_size, augment=True
    )
    val_dataset = FaceParsingDataset(
        img_dir=os.path.join(args.data_dir, 'val', 'images'),
        mask_dir=os.path.join(args.data_dir, 'val', 'masks'),
        img_size=args.img_size, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(args.arch, args.num_classes, args.base_ch).to(device)
    criterion = CombinedLoss(num_classes=args.num_classes)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = os.path.join(args.save_dir, args.arch)
    os.makedirs(save_dir, exist_ok=True)

    best_f_score = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, f_score = validate(model, val_loader, criterion, device, args.num_classes)
        scheduler.step()

        print(f"[{args.arch}] Epoch {epoch:03d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | F: {f_score:.4f}")

        if f_score > best_f_score:
            best_f_score = f_score
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'f_score': f_score,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  >>> Saved best (F={f_score:.4f})")

    print(f"\nDone. Best F-score [{args.arch}]: {best_f_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='attention_unet',
                        choices=['attention_unet', 'deeplab', 'segnet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
