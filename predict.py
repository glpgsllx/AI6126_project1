import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def get_model(arch, num_classes, base_ch):
    if arch == 'attention_unet':
        from src.model import AttentionUNet
        return AttentionUNet(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    if arch == 'deeplab':
        from src.model_b_deeplab import LightDeepLab
        return LightDeepLab(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    if arch == 'segnet':
        from src.model_c_segnet import LightSegNet
        return LightSegNet(in_ch=3, num_classes=num_classes, base_ch=base_ch)
    raise ValueError(f"Unknown arch: {arch}")


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = get_model(args.arch, args.num_classes, args.base_ch)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_arch = checkpoint.get('arch')
    if ckpt_arch is not None and ckpt_arch != args.arch:
        raise ValueError(f"Checkpoint arch is '{ckpt_arch}', but --arch is '{args.arch}'.")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(
        f"Loaded {args.arch} checkpoint "
        f"(epoch {checkpoint['epoch']}, F-score {checkpoint['f_score']:.4f})"
    )

    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(args.test_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    print(f"Running inference on {len(img_files)} images...")

    with torch.no_grad():
        for img_name in img_files:
            img_path = os.path.join(args.test_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            orig_size = image.size  # (W, H)

            inp = transform(image).unsqueeze(0).to(device)
            logits = model(inp)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Resize back to original size
            pred_img = Image.fromarray(pred)
            pred_img = pred_img.resize(orig_size, Image.NEAREST)

            base_name = os.path.splitext(img_name)[0]
            out_path = os.path.join(args.output_dir, base_name + '.png')
            pred_img.save(out_path)

    print(f"Predictions saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='attention_unet',
                        choices=['attention_unet', 'deeplab', 'segnet'])
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/attention_unet/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='./predictions')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--base_ch', type=int, default=15)
    args = parser.parse_args()
    predict(args)
