import os
import io
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class FaceParsingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, augment=False, num_classes=19, ignore_index=255):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]

        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load mask — try .png first, then .jpg
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(mask_path):
                break
        # Keep indexed class ids from palette masks; do not convert to grayscale.
        mask = Image.open(mask_path)

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Augmentation
        if self.augment:
            image, mask = self._augment(image, mask)

        # To tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        mask_np = np.array(mask, dtype=np.int64)
        invalid = (mask_np < 0) | (mask_np >= self.num_classes)
        if invalid.any():
            mask_np[invalid] = self.ignore_index
        mask = torch.from_numpy(mask_np).long()

        return image, mask

    def _augment(self, image, mask):
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            # Optional variant (tested but not selected):
            # mask = TF.rotate(
            #     mask,
            #     angle,
            #     interpolation=Image.NEAREST,
            #     fill=self.ignore_index,
            # )
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        # Random crop and resize
        if random.random() > 0.5:
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1) # image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, h, w,
                                    (self.img_size, self.img_size), Image.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w,
                                   (self.img_size, self.img_size), Image.NEAREST)

        # Color jitter (image only)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))

        # Weak Gaussian blur (image only)
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))

        # Weak JPEG compression artifact simulation (image only)
        if random.random() < 0.2:
            image = self._jpeg_compress(image, quality_min=70, quality_max=95)

        # Random noise cutout (image only): 1 hole, 4%-6% area
        if random.random() < 0.2:
            image = self._random_noise_cutout(image, min_area_ratio=0.04, max_area_ratio=0.06)

        return image, mask

    def _random_noise_cutout(self, image, min_area_ratio=0.04, max_area_ratio=0.06):
        w, h = image.size
        img_np = np.array(image, dtype=np.uint8)

        area = w * h
        target_area = random.uniform(min_area_ratio, max_area_ratio) * area
        aspect = random.uniform(0.75, 1.33)
        hole_w = int(round((target_area * aspect) ** 0.5))
        hole_h = int(round((target_area / max(aspect, 1e-6)) ** 0.5))
        hole_w = max(1, min(hole_w, w))
        hole_h = max(1, min(hole_h, h))

        x0 = random.randint(0, w - hole_w)
        y0 = random.randint(0, h - hole_h)
        noise = np.random.randint(0, 256, size=(hole_h, hole_w, 3), dtype=np.uint8)
        img_np[y0:y0 + hole_h, x0:x0 + hole_w] = noise
        return Image.fromarray(img_np)

    def _jpeg_compress(self, image, quality_min=70, quality_max=95):
        quality = random.randint(quality_min, quality_max)
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
