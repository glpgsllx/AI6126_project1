import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    """Depthwise Separable Conv -> BN -> ReLU (x2)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # first DS conv
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # second DS conv
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightSegNet(nn.Module):
    """
    SegNet-style encoder-decoder with depthwise separable convolutions.
    Uses MaxPool indices for upsampling (no extra params for decoder upsample).
    Much more parameter-efficient than standard SegNet.
    """
    def __init__(self, in_ch=3, num_classes=19, base_ch=32):
        super().__init__()
        b = base_ch

        # Encoder
        self.enc1 = DSConvBlock(in_ch, b)       # 32
        self.enc2 = DSConvBlock(b, b * 2)        # 64
        self.enc3 = DSConvBlock(b * 2, b * 4)   # 128
        self.enc4 = DSConvBlock(b * 4, b * 8)   # 256

        self.pool = nn.MaxPool2d(2, return_indices=True)

        # Bottleneck
        self.bottleneck = DSConvBlock(b * 8, b * 8)

        # Decoder
        self.unpool = nn.MaxUnpool2d(2)

        self.dec4 = DSConvBlock(b * 8, b * 4)
        self.dec3 = DSConvBlock(b * 4, b * 2)
        self.dec2 = DSConvBlock(b * 2, b)
        self.dec1 = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1p, idx1 = self.pool(e1)

        e2 = self.enc2(e1p)
        e2p, idx2 = self.pool(e2)

        e3 = self.enc3(e2p)
        e3p, idx3 = self.pool(e3)

        e4 = self.enc4(e3p)
        e4p, idx4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(e4p)

        # Decoder
        d4 = self.unpool(b, idx4)
        d4 = self.dec4(d4)

        d3 = self.unpool(d4, idx3)
        d3 = self.dec3(d3)

        d2 = self.unpool(d3, idx2)
        d2 = self.dec2(d2)

        d1 = self.unpool(d2, idx1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = LightSegNet(in_ch=3, num_classes=19, base_ch=32)
    total = count_parameters(model)
    print(f"LightSegNet parameters: {total:,}")
    print(f"Limit: 1,821,085")
    print(f"Within limit: {total < 1821085}")

    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape: {out.shape}")
