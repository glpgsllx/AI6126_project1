import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """标准的 3x3 深度可分离卷积"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlockDW(nn.Module):
    """升级版 Double Conv: 将普通卷积替换为极其省参数的 DW 卷积"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 为了保证第一层特征提取能力，如果输入是 3 通道(原图)，我们保留一个标准卷积
        if in_ch == 3:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(out_ch, out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                DepthwiseSeparableConv(in_ch, out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(out_ch, out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention Gate 保持不变，它用的是 1x1 卷积，参数很少"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    # 【高能预警】base_ch 默认值提到了 24！(甚至可以尝试 26)
    def __init__(self, in_ch=3, num_classes=19, base_ch=24):
        super().__init__()

        # Encoder (全部换成了省参数的 ConvBlockDW)
        self.enc1 = ConvBlockDW(in_ch, base_ch)        
        self.enc2 = ConvBlockDW(base_ch, base_ch*2)    
        self.enc3 = ConvBlockDW(base_ch*2, base_ch*4)  
        self.enc4 = ConvBlockDW(base_ch*4, base_ch*8)  

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlockDW(base_ch*8, base_ch*16)  

        # Attention gates
        self.att4 = AttentionGate(F_g=base_ch*8, F_l=base_ch*8, F_int=base_ch*4)
        self.att3 = AttentionGate(F_g=base_ch*4, F_l=base_ch*4, F_int=base_ch*2)
        self.att2 = AttentionGate(F_g=base_ch*2, F_l=base_ch*2, F_int=base_ch)
        self.att1 = AttentionGate(F_g=base_ch,   F_l=base_ch,   F_int=max(1, base_ch//2))

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = ConvBlockDW(base_ch*16, base_ch*8)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlockDW(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlockDW(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBlockDW(base_ch*2, base_ch)

        # Output
        self.out_conv = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    # 你可以试试把 base_ch 设为 24, 26 甚至 28，看看哪个最逼近 182 万
    model = AttentionUNet(in_ch=3, num_classes=19, base_ch=24)
    total = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Limit: 1,821,085")
    print(f"Within limit: {total < 1821085}")

    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape: {out.shape}")  #[2, 19, 512, 512]