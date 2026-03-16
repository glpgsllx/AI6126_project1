import os
import argparse
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image

from src.dataset import FaceParsingDataset
from src.losses import CombinedLoss


def parse_epoch_list(text):
    if text is None or str(text).strip() == '':
        return set()
    epochs = set()
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"--save_epochs only accepts positive integers, got: {value}")
        epochs.add(value)
    return epochs


def save_checkpoint(path, epoch, arch, model, f_score):
    torch.save({
        'epoch': epoch,
        'arch': arch,
        'model_state_dict': model.state_dict(),
        'f_score': f_score,
    }, path)


def compute_class_weights(mask_dir, num_classes, scheme="median_freq", transform="none", clip_max=0.0):
    counts = np.zeros(num_classes, dtype=np.int64)
    for file_name in os.listdir(mask_dir):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, file_name)), dtype=np.int64)
        vals, cnts = np.unique(mask, return_counts=True)
        for val, cnt in zip(vals.tolist(), cnts.tolist()):
            if 0 <= val < num_classes:
                counts[val] += cnt

    freqs = counts / max(counts.sum(), 1)
    positive = freqs > 0
    weights = np.ones(num_classes, dtype=np.float32)

    if scheme == "median_freq":
        median_freq = np.median(freqs[positive])
        weights[positive] = (median_freq / freqs[positive]).astype(np.float32)
    else:
        raise ValueError(f"Unknown class weight scheme: {scheme}")

    if transform == "sqrt":
        weights = np.sqrt(weights).astype(np.float32)
    elif transform != "none":
        raise ValueError(f"Unknown class weight transform: {transform}")

    if clip_max > 0:
        weights = np.minimum(weights, clip_max).astype(np.float32)

    return torch.tensor(weights, dtype=torch.float32)


def save_metrics_csv(csv_path, history):
    fields = ['epoch', 'lr', 'train_loss', 'val_loss', 'f_score']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(history['epoch'])):
            writer.writerow({
                'epoch': history['epoch'][i],
                'lr': history['lr'][i],
                'train_loss': history['train_loss'][i],
                'val_loss': history['val_loss'][i],
                'f_score': history['f_score'][i],
            })


def save_metrics_plot(plot_path, history):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not found, skipped metrics plot.")
        return

    epochs = history['epoch']
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='train_loss')
    if any(v is not None for v in history['val_loss']):
        val_vals = [float('nan') if v is None else v for v in history['val_loss']]
        plt.plot(epochs, val_vals, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    if any(v is not None for v in history['f_score']):
        f_vals = [float('nan') if v is None else v for v in history['f_score']]
        plt.plot(epochs, f_vals, label='f_score')
    plt.xlabel('epoch')
    plt.ylabel('f-score')
    plt.title('F-score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_model(arch, num_classes, base_ch):
    if arch == 'attention_unet':
        from src.model import AttentionUNet, count_parameters
        model = AttentionUNet(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    elif arch == 'deeplab':
        from src.model_b_deeplab import LightDeepLab, count_parameters
        model = LightDeepLab(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    elif arch == 'deeplab_aspp4812':
        from src.model_b_deeplab_aspp4812 import LightDeepLabASPP4812, count_parameters
        model = LightDeepLabASPP4812(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    elif arch == 'deeplab_dwstage3':
        from src.model_b_deeplab_dwstage3 import LightDeepLabDWStage3, count_parameters
        model = LightDeepLabDWStage3(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    elif arch == 'segnet':
        from src.model_c_segnet import LightSegNet, count_parameters
        model = LightSegNet(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    total = sum(p.numel() for p in model.parameters())
    print(f"[{arch}] Parameters: {total:,} / 1,821,085 limit")
    assert total < 1821085, f"Parameter limit exceeded! ({total:,})"
    return model


def compute_f_measure(preds, targets, num_classes, ignore_index=255, beta=1.0):
    f_scores = []
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    if targets.size == 0:
        return 0.0

    # Match course metric: average F-score over classes present in GT.
    for cls in np.unique(targets):
        cls = int(cls)
        tp = ((preds == cls) & (targets == cls)).sum()
        fp = ((preds == cls) & (targets != cls)).sum()
        fn = ((preds != cls) & (targets == cls)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        beta2 = beta ** 2
        f = (1 + beta2) * (precision * recall) / (beta2 * precision + recall + 1e-8)
        f_scores.append(f)

    return float(np.mean(f_scores)) if f_scores else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device, log_interval=0, arch='model', epoch=1, total_epochs=1):
    model.train()
    total_loss = 0
    num_batches = len(loader)
    t0 = time.time()
    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if log_interval > 0 and (step % log_interval == 0 or step == num_batches):
            elapsed = time.time() - t0
            avg = total_loss / step
            print(
                f"[{arch}] Epoch {epoch:03d}/{total_epochs} | "
                f"Step {step:04d}/{num_batches:04d} | "
                f"Loss: {loss.item():.4f} | Avg: {avg:.4f} | {elapsed:.1f}s",
                flush=True
            )

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
    device = select_device()
    print(f"Device: {device} | Arch: {args.arch}")

    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if not (0.0 <= args.val_split < 1.0):
        raise ValueError(f"--val_split must be in [0, 1), got {args.val_split}")

    train_img_dir = os.path.join(args.data_dir, 'train', 'images')
    train_mask_dir = os.path.join(args.data_dir, 'train', 'masks')

    train_dataset = FaceParsingDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        img_size=args.img_size, augment=True,
        num_classes=args.num_classes, ignore_index=255
    )
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = args.persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    val_loader = None
    if args.val_split > 0.0:
        full_train_aug = train_dataset
        full_train_noaug = FaceParsingDataset(
            img_dir=train_img_dir,
            mask_dir=train_mask_dir,
            img_size=args.img_size, augment=False,
            num_classes=args.num_classes, ignore_index=255
        )
        n = len(full_train_aug)
        val_size = max(1, int(n * args.val_split))
        train_size = n - val_size
        if train_size <= 0:
            raise ValueError(f"Train split is empty. Reduce --val_split (current {args.val_split}).")

        g = torch.Generator()
        g.manual_seed(args.split_seed)
        perm = torch.randperm(n, generator=g).tolist()
        train_idx = perm[:train_size]
        val_idx = perm[train_size:]

        train_loader = DataLoader(Subset(full_train_aug, train_idx), shuffle=True, **loader_kwargs)
        val_loader = DataLoader(Subset(full_train_noaug, val_idx), shuffle=False, **loader_kwargs)
        print(f"Using train split for validation: train={train_size}, val={val_size}, seed={args.split_seed}")
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_mask_dir = os.path.join(args.data_dir, 'val', 'masks')
        if os.path.isdir(val_mask_dir):
            val_dataset = FaceParsingDataset(
                img_dir=os.path.join(args.data_dir, 'val', 'images'),
                mask_dir=val_mask_dir,
                img_size=args.img_size, augment=False,
                num_classes=args.num_classes, ignore_index=255
            )
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
            print("Using data/val with masks for validation.")
        else:
            print("No val masks found. Training without validation metrics.")

    model = get_model(args.arch, args.num_classes, args.base_ch).to(device)
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled (mode={args.compile_mode})")
        except Exception as e:
            print(f"torch.compile unavailable, fallback to eager mode: {e}")

    ce_class_weights = None
    if args.weighted_ce != 'none':
        ce_class_weights = compute_class_weights(
            train_mask_dir,
            args.num_classes,
            scheme=args.weighted_ce,
            transform=args.weighted_ce_transform,
            clip_max=args.weighted_ce_clip_max,
        ).to(device)
        print(f"Using weighted CE: {args.weighted_ce}")
        print(f"Weight transform: {args.weighted_ce_transform} | clip_max: {args.weighted_ce_clip_max}")
        print(f"CE class weights: {[round(float(x), 4) for x in ce_class_weights.cpu().tolist()]}")
        if args.ce_type == 'lovasz':
            print("Note: ce_type=lovasz ignores CE class weights and boundary CE settings.")

    criterion = CombinedLoss(
        num_classes=args.num_classes,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        ce_class_weights=ce_class_weights,
        boundary_ce_factor=args.boundary_ce_factor if args.ce_type == 'ce' else 0.0,
        ce_type=args.ce_type,
        focal_gamma=args.focal_gamma,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = os.path.join(args.save_dir, args.arch)
    if args.exp_name:
        save_dir = os.path.join(save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_epochs = parse_epoch_list(args.save_epochs)
    metrics_csv_path = os.path.join(save_dir, 'metrics.csv')
    metrics_plot_path = os.path.join(save_dir, 'metrics.png')

    history = {
        'epoch': [],
        'lr': [],
        'train_loss': [],
        'val_loss': [],
        'f_score': [],
    }

    best_f_score = 0.0
    best_train_loss = float('inf')
    epochs_without_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            log_interval=args.log_interval, arch=args.arch, epoch=epoch, total_epochs=args.epochs
        )
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        val_loss = None
        f_score = None

        if val_loader is not None:
            val_loss, f_score = validate(model, val_loader, criterion, device, args.num_classes)
            print(f"[{args.arch}] Epoch {epoch:03d}/{args.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | F: {f_score:.4f} | LR: {lr_now:.2e}")

            if f_score > best_f_score:
                best_f_score = f_score
                epochs_without_improve = 0
                save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch,
                    args.arch,
                    model,
                    f_score,
                )
                print(f"  >>> Saved best (F={f_score:.4f})")
            else:
                epochs_without_improve += 1
        else:
            print(f"[{args.arch}] Epoch {epoch:03d}/{args.epochs} | Train: {train_loss:.4f} | LR: {lr_now:.2e}")
            if args.save_best_train_loss and train_loss < best_train_loss:
                best_train_loss = train_loss
                save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch,
                    args.arch,
                    model,
                    -1.0,
                )
                print(f"  >>> Saved best by train loss ({train_loss:.4f})")

        history['epoch'].append(epoch)
        history['lr'].append(lr_now)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['f_score'].append(f_score)
        save_metrics_csv(metrics_csv_path, history)

        if epoch in save_epochs:
            ckpt_path = os.path.join(save_dir, f'epoch_{epoch:03d}.pth')
            save_checkpoint(
                ckpt_path,
                epoch,
                args.arch,
                model,
                -1.0 if f_score is None else f_score,
            )
            print(f"  >>> Saved requested checkpoint: {ckpt_path}")

        if val_loader is not None and args.early_stop_patience > 0:
            if epochs_without_improve >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}: no F-score improvement for "
                    f"{args.early_stop_patience} epoch(s)."
                )
                break

    save_metrics_plot(metrics_plot_path, history)
    if val_loader is None and not args.save_best_train_loss and args.save_last:
        last_path = os.path.join(save_dir, 'last_model.pth')
        save_checkpoint(last_path, args.epochs, args.arch, model, -1.0)
        print(f"Saved last checkpoint: {last_path}")
    print(f"Saved metrics csv: {metrics_csv_path}")
    print(f"Saved metrics plot: {metrics_plot_path}")

    if val_loader is not None:
        print(f"\nDone. Best F-score [{args.arch}]: {best_f_score:.4f}")
    else:
        print(f"\nDone. Best train loss [{args.arch}]: {best_train_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='attention_unet',
                        choices=['attention_unet', 'deeplab', 'deeplab_aspp4812', 'deeplab_dwstage3', 'segnet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--base_ch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    parser.add_argument('--no_persistent_workers', action='store_false', dest='persistent_workers')
    parser.add_argument('--compile', action='store_true', default=True)
    parser.add_argument('--no_compile', action='store_false', dest='compile')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--matmul_precision', type=str, default='high',
                        choices=['highest', 'high', 'medium'])
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--val_split', type=float, default=0.0,
                        help='If >0, split this fraction from train set as validation.')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for train/val split when --val_split > 0.')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop early after this many validation epochs without F-score improvement. 0 disables.')
    parser.add_argument('--exp_name', type=str, default='',
                        help='Optional experiment name. Saves outputs under save_dir/arch/exp_name.')
    parser.add_argument('--weighted_ce', type=str, default='none',
                        choices=['none', 'median_freq'],
                        help='Optional class weighting scheme for CrossEntropyLoss.')
    parser.add_argument('--weighted_ce_transform', type=str, default='none',
                        choices=['none', 'sqrt'],
                        help='Optional transform applied to CE class weights.')
    parser.add_argument('--weighted_ce_clip_max', type=float, default=0.0,
                        help='If > 0, clip CE class weights to this maximum value.')
    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--boundary_ce_factor', type=float, default=0.0,
                        help='Extra CE weight applied to boundary pixels. 0 disables boundary-weighted CE.')
    parser.add_argument('--ce_type', type=str, default='ce',
                        choices=['ce', 'focal', 'lovasz'],
                        help='Type of classification loss combined with Dice.')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma used when ce_type=focal.')
    parser.add_argument('--save_epochs', type=str, default='',
                        help='Comma-separated epoch numbers to save extra checkpoints, e.g. "41,45".')
    parser.add_argument('--save_best_train_loss', action='store_true', default=False,
                        help='When no validation metrics are available, keep updating best_model.pth by train loss.')
    parser.add_argument('--save_last', action='store_true', default=True,
                        help='When no validation metrics are available, save last_model.pth at training end.')
    parser.add_argument('--no_save_last', action='store_false', dest='save_last')
    args = parser.parse_args()
    main(args)
