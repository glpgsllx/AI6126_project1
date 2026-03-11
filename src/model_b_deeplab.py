import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                   padding=dilation, dilation=dilation,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.gap(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.project(torch.cat([x1, x2, x3, x4, x5], dim=1))


class LightDeepLab(nn.Module):
    """
    Lightweight DeepLabV3+ style.
    Encoder: stride conv + depthwise separable + dilated residual blocks
    Decoder: ASPP + skip connection + upsample
    """
    def __init__(self, in_ch=3, num_classes=19, base_ch=32):
        super().__init__()
        b = base_ch

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        )  # /2

        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(b, b * 2, stride=2),
        )  # /4

        self.low_proj = nn.Sequential(
            nn.Conv2d(b * 2, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )

        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(b * 2, b * 4, stride=2),
            ResidualBlock(b * 4, dilation=1),
            ResidualBlock(b * 4, dilation=2),
            ResidualBlock(b * 4, dilation=4),
        )  # /8

        self.aspp = ASPP(b * 4, b * 4)

        self.decoder = nn.Sequential(
            nn.Conv2d(b * 4 + 32, b * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
            DepthwiseSeparableConv(b * 2, b * 2),
            nn.Conv2d(b * 2, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        low = self.low_proj(s2)
        s3 = self.stage3(s2)

        feat = self.aspp(s3)
        feat = F.interpolate(feat, size=low.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([feat, low], dim=1)
        out = self.decoder(feat)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = LightDeepLab(in_ch=3, num_classes=19, base_ch=32)
    total = count_parameters(model)
    print(f"LightDeepLab parameters: {total:,}")
    print(f"Limit: 1,821,085")
    print(f"Within limit: {total < 1821085}")

    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape: {out.shape}")
