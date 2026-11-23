"""Matched lightweight U-Nets; only the optional bottleneck block differs."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels), nn.ReLU(inplace=True))

def position_encoding(height, width, channels):
    if channels % 4: raise ValueError('Position channels must divide by four')
    omega = 1 / (10000 ** (torch.arange(channels//4, dtype=torch.float32)/(channels//4)))
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    x, y = x.reshape(-1, 1)*omega, y.reshape(-1, 1)*omega
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1).unsqueeze(0)

class SpatialAttention(nn.Module):
    def __init__(self, channels, heads=4, dropout=0.1):
        super().__init__()
        self.register_buffer('position', position_encoding(16,16,channels))
        self.block = nn.TransformerEncoderLayer(channels, heads, channels*2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)

    def forward(self, x):
        b,c,h,w = x.shape
        if (h,w) != (16,16): raise ValueError('This protocol requires 256x256 inputs')
        tokens = x.flatten(2).transpose(1,2)
        tokens = self.block(tokens + self.position.to(dtype=x.dtype))
        return tokens.transpose(1,2).reshape(b,c,h,w)

class UNet(nn.Module):
    def __init__(self, attention=False, width=16):
        super().__init__()
        widths = [width*2**i for i in range(5)]
        self.encoder = nn.ModuleList([DoubleConv(6 if i == 0 else widths[i-1], w)
                                     for i,w in enumerate(widths)])
        self.decoder = nn.ModuleList([DoubleConv(widths[i+1]+widths[i], widths[i])
                                     for i in range(3,-1,-1)])
        self.output = nn.Conv2d(width, 3, 1)
        # Initialize common layers first so a matched seed has identical common weights.
        self.attention = SpatialAttention(widths[-1]) if attention else nn.Identity()

    def forward(self, x):
        features = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        x = self.attention(x)
        for decoder, skip in zip(self.decoder, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.output(x)

def segmentation_loss(logits, target):
    ce = F.cross_entropy(logits, target)
    probs = logits.softmax(1)
    onehot = F.one_hot(target, 3).permute(0,3,1,2).to(probs.dtype)
    p, t = probs[:,1:], onehot[:,1:]
    dims = (0,2,3)
    # Batch-pooled Dice; absent classes retain the denominator to penalize FP.
    dice = (2*(p*t).sum(dims)+1e-6)/(p.sum(dims)+t.sum(dims)+1e-6)
    return .5*ce + .5*(1-dice.mean())
