import os
import argparse
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.dataset import FaceParsingDataset
from src.losses import CombinedLoss


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
    elif arch == 'segnet':
        from src.model_c_segnet import LightSegNet, count_parameters
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

    train_dataset = FaceParsingDataset(
        img_dir=os.path.join(args.data_dir, 'train', 'images'),
        mask_dir=os.path.join(args.data_dir, 'train', 'masks'),
        img_size=args.img_size, augment=True
    )
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = args.persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    val_mask_dir = os.path.join(args.data_dir, 'val', 'masks')
    val_loader = None
    if os.path.isdir(val_mask_dir):
        val_dataset = FaceParsingDataset(
            img_dir=os.path.join(args.data_dir, 'val', 'images'),
            mask_dir=val_mask_dir,
            img_size=args.img_size, augment=False
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    else:
        print("No val masks found. Training without validation metrics.")

    model = get_model(args.arch, args.num_classes, args.base_ch).to(device)
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled (mode={args.compile_mode})")
        except Exception as e:
            print(f"torch.compile unavailable, fallback to eager mode: {e}")

    criterion = CombinedLoss(num_classes=args.num_classes)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = os.path.join(args.save_dir, args.arch)
    os.makedirs(save_dir, exist_ok=True)
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
                torch.save({
                    'epoch': epoch,
                    'arch': args.arch,
                    'model_state_dict': model.state_dict(),
                    'f_score': f_score,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  >>> Saved best (F={f_score:.4f})")
        else:
            print(f"[{args.arch}] Epoch {epoch:03d}/{args.epochs} | Train: {train_loss:.4f} | LR: {lr_now:.2e}")
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'arch': args.arch,
                    'model_state_dict': model.state_dict(),
                    'f_score': -1.0,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  >>> Saved best by train loss ({train_loss:.4f})")

        history['epoch'].append(epoch)
        history['lr'].append(lr_now)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['f_score'].append(f_score)
        save_metrics_csv(metrics_csv_path, history)

    save_metrics_plot(metrics_plot_path, history)
    print(f"Saved metrics csv: {metrics_csv_path}")
    print(f"Saved metrics plot: {metrics_plot_path}")

    if val_loader is not None:
        print(f"\nDone. Best F-score [{args.arch}]: {best_f_score:.4f}")
    else:
        print(f"\nDone. Best train loss [{args.arch}]: {best_train_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='attention_unet',
                        choices=['attention_unet', 'deeplab', 'segnet'])
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
    args = parser.parse_args()
    main(args)
