import torch
from torch import nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_bn=True, use_gelu=True, use_residual=True):
        super(ConvBlock3D, self).__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size, stride, padding)
        if use_gelu:
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
        if use_bn:
            self.norm1 = nn.BatchNorm3d(out_ch)
            self.norm2 = nn.BatchNorm3d(out_ch)
            if in_ch != out_ch:
                self.shortcut = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(out_ch))
            else:
                self.shortcut = nn.Identity()
        else:
            self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            if in_ch != out_ch:
                self.shortcut = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0), nn.GroupNorm(min(8, out_ch), out_ch))
            else:
                self.shortcut = nn.Identity()


    def forward(self, x):
        if self.use_residual:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.use_residual:
            out += residual
        out = self.act(out)
        return out

class Downsample3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Downsample3D,self).__init__()
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch)
        
    def forward(self, x):
        out = self.down(x)
        out = self.conv(out)
        return out

class Encoder3D(nn.Module):
    def __init__(self, in_ch, base_ch=32, num_layers=4):
        super(Encoder3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            out_ch = base_ch * (2 ** i)
            if i == 0:
                self.layers.append(ConvBlock3D(in_ch, out_ch))
            else:
                self.layers.append(Downsample3D(base_ch * (2 ** (i - 1)), out_ch))

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features
    
class Upsample3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upsample3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch)
    
    def forward(self, x, skip):
        out = self.up(x)
        out = torch.cat((out, skip), dim=1)
        out = self.conv(out)
        return out

class Decoder3D(nn.Module):
    def __init__(self, out_ch, base_ch=32, num_layers=4):
        super(Decoder3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers-2, -1, -1):
            ch = base_ch * (2 ** i)
            self.layers.append(Upsample3D(ch * 2, ch))
        self.out_conv = nn.Conv3d(base_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        for i, layer in enumerate(self.layers):
            x = layer(features[-1] if i == 0 else x, features[-(i+2)])
        out = self.out_conv(x)
        return out

class DeepONet3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, num_layers=4):
        super(DeepONet3D, self).__init__()
        self.trunk = Encoder3D(in_ch, base_ch, num_layers)
        self.branch = Encoder3D(in_ch, base_ch, num_layers)
        self.decoder = Decoder3D(out_ch, base_ch, num_layers)

    def forward(self, E, r):
        trunk_features = self.trunk(r)
        branch_features = self.branch(E)
        features = [tf * bf for tf, bf in zip(trunk_features, branch_features)]
        out = self.decoder(features)
        return out
        