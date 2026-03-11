import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from predict import get_model, get_palette


def overlay_mask(image, mask, alpha=0.4):
    palette = np.array(get_palette(), dtype=np.uint8).reshape(256, 3)
    color_mask = palette[mask]
    image_np = np.array(image, dtype=np.uint8)
    blended = ((1 - alpha) * image_np + alpha * color_mask).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def infer_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.arch, args.num_classes, args.base_ch)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_arch = checkpoint.get("arch")
    if ckpt_arch is not None and ckpt_arch != args.arch:
        raise ValueError(f"Checkpoint arch is '{ckpt_arch}', but --arch is '{args.arch}'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    orig_size = image.size

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        inp = transform(image).unsqueeze(0).to(device)
        logits = model(inp)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_img = Image.fromarray(pred).resize(orig_size, Image.NEAREST)
    pred_np = np.array(pred_img, dtype=np.uint8)
    pred_img.putpalette(get_palette())
    overlay_img = overlay_mask(image, pred_np, alpha=args.alpha)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    pred_path = os.path.join(args.output_dir, f"{base_name}_pred.png")
    overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")

    pred_img.save(pred_path)
    overlay_img.save(overlay_path)

    print(f"Saved prediction: {pred_path}")
    print(f"Saved overlay: {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="/home/work/ldv-fs-sg/chenyixuan/seg/data/val/images/0a4fdac4e7f448718c42f34a5422aee2.jpg", type=str)
    parser.add_argument("--checkpoint", default="/home/work/ldv-fs-sg/chenyixuan/seg/checkpoints/deeplab_d7c3/best_model.pth", type=str)
    parser.add_argument("--arch", type=str, default="deeplab",
                        choices=["attention_unet", "deeplab", "segnet"])
    parser.add_argument("--output_dir", type=str, default="./visualizations")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()
    infer_single(args)
