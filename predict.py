import os
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from model import AttentionUNet


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = AttentionUNet(in_ch=3, num_classes=args.num_classes, base_ch=args.base_ch)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, F-score {checkpoint['f_score']:.4f})")

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
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='./predictions')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--base_ch', type=int, default=32)
    args = parser.parse_args()
    predict(args)
