import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class FaceParsingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

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
        mask = Image.open(mask_path).convert('L')

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
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

    def _augment(self, image, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        # Random crop and resize
        if random.random() > 0.5:
            i, j, h, w = TF.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
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

        return image, mask
