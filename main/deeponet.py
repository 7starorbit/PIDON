import torch
from torch import nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_ch)
        self.gelu = nn.GELU()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(out_ch))
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.gelu(out)
        out = self.conv(out)
        out = self.bn(out)
        out += residual
        out = self.gelu(out)
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
        features.append(layer(x) for layer in self.layers)
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
            out_ch = base_ch * (2 ** i)
            self.layers.append(Upsample3D(out_ch * 2, out_ch))
        self.out_conv = nn.Conv3d(base_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        for i, layer in enumerate(self.layers):
            x = layer(features[-1] if i == 0 else x, features[-(i+2)])
        out = self.out_conv(x)
        return out

class DeepONet3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, num_layers=4):
        super(DeepONet3D, self).__init__()
        self.encoder = Encoder3D(in_ch, base_ch, num_layers)
        self.decoder = Decoder3D(out_ch, base_ch, num_layers)

    def forward(self, E, r):
        trunk_features = self.encoder(r)
        branch_features = self.encoder(E)
        features = [tf * bf for tf, bf in zip(trunk_features, branch_features)]
        out = self.decoder(features)
        return out
        