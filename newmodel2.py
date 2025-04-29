import torch
import torch.nn as nn
import torch.nn.functional as F

class MagUNet(nn.Module):
    def __init__(self):
        super(MagUNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(2, 64) # Input (2, 1028, 512)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = conv_block(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = conv_block(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, 8, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        b = self.bottleneck(p2)

        u1 = self.up1(b)
        u1 = torch.cat([u1, c2], dim=1)
        c4 = self.dec1(u1)

        u2 = self.up2(c4)
        u2 = torch.cat([u2, c1], dim=1)
        c5 = self.dec2(u2)

        outputs = self.final_conv(c5)
        outputs = outputs.view(-1, 4, 2, 1028, 512)

        return outputs
    
class NewUNet(nn.Module):
    def __init__(self):
        super(NewUNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout=0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            )
        
        # (-, 2, 1024, 512)
        self.enc1 = conv_block(2, 64) # (-, 64, 1024, 512)
        self.pool1 = nn.MaxPool2d(2) # (-, 64, 512, 256)

        self.enc2 = conv_block(64, 128) # (-, 128, 512, 256)
        self.pool2 = nn.MaxPool2d(2) # (-, 128, 256, 128)

        self.enc3 = conv_block(128, 256, 0.1) # (-, 256, 256, 128)
        self.pool3 = nn.MaxPool2d(2) # (-, 256, 128, 64)

        self.bottleneck = conv_block(256, 512, 0.2) # (-, 512, 128, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # (-, 512, 256, 128)
        self.dec1 = conv_block(512 + 256, 256, 0.1) # (-, 256, 256, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # (-, 256, 512, 256)
        self.dec2 = conv_block(256 + 128, 128) # (-, 128, 512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # (-, 128, 1024, 512)
        self.dec3 = conv_block(128 + 64, 64) # (-, 64, 1024, 512)

        self.final_conv = nn.Conv2d(64, 4*2, kernel_size=1) # (-, 8, 1024, 512)

    def forward(self, x):
        c1 = self.enc1(x) # 128MB
        p1 = self.pool1(c1) # 32MB

        c2 = self.enc2(p1) # 64MB
        p2 = self.pool2(c2) # 16MB

        c3 = self.enc3(p2) # 32MB
        p3 = self.pool3(c3) # 8MB

        c4 = self.bottleneck(p3) # 16MB

        u1 = self.up1(c4)
        u1 = torch.cat([u1, c3], dim=1) # 64MB + 32MB
        c5 = self.dec1(u1) # 32MB

        u2 = self.up2(c5)
        u2 = torch.cat([u2, c2], dim=1) # 128MB + 64MB
        c6 = self.dec2(u2) # 64MB

        u3 = self.up3(c6)
        u3 = torch.cat([u3, c1], dim=1) # 256MB + 128MB
        c7 = self.dec3(u3) # 128MB

        outputs = self.final_conv(c7) # 16MB
        outputs = outputs.reshape(-1, 4, 2, 1024, 512)
        return outputs
    
class NewUNet2(nn.Module): # v1  dropout: 0, 0, 0.1, 0.1, 0.2, 0.3 | v2 dropout: 0, 0, 0, 0, 0, 0
    def __init__(self):
        super(NewUNet2, self).__init__()

        def conv_block(in_channels, out_channels, dropout=0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            )
        
        # (-, 2, 512, 512)
        self.enc1 = conv_block(2, 64) # (-, 64, 512, 512)
        self.pool1 = nn.MaxPool2d(2) # (-, 64, 256, 256)

        self.enc2 = conv_block(64, 128) # (-, 128, 256, 256)
        self.pool2 = nn.MaxPool2d(2) # (-, 128, 128, 128)

        self.enc3 = conv_block(128, 256) # (-, 256, 128, 128)
        self.pool3 = nn.MaxPool2d(2) # (-, 256, 64, 64)

        self.enc4 = conv_block(256, 512) # (-, 512, 64, 64)
        self.pool4 = nn.MaxPool2d(2) # (-, 512, 32, 32)

        self.enc5 = conv_block(512, 1024) # (-, 1024, 32, 32)
        self.pool5 = nn.MaxPool2d(2) # (-, 1024, 16, 16)

        self.bottleneck = conv_block(1024, 2048) # (-, 2048, 16, 16)

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2) # (-, 2048, 32, 32)
        self.dec1 = conv_block(2048, 1024) # (-, 1024, 32, 32)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # (-, 1024, 64, 64)
        self.dec2 = conv_block(1024, 512) # (-, 512, 64, 64)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # (-, 512, 128, 128)
        self.dec3 = conv_block(512, 256) # (-, 256, 128, 128)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # (-, 256, 256, 256)
        self.dec4 = conv_block(256, 128) # (-, 128, 512, 256)

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # (-, 128, 512, 512)
        self.dec5 = conv_block(128, 64) # (-, 64, 512, 512)

        self.final_conv = nn.Conv2d(64, 4*2, kernel_size=1) # (-, 8, 512, 512)

    def forward(self, x):
        c1 = self.enc1(x) 
        p1 = self.pool1(c1) 

        c2 = self.enc2(p1) 
        p2 = self.pool2(c2) 

        c3 = self.enc3(p2) 
        p3 = self.pool3(c3) 

        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        c5 = self.enc5(p4)
        p5 = self.pool5(c5)

        c6 = self.bottleneck(p5) 

        u1 = self.up1(c6)
        u1 = torch.cat([u1, c5], dim=1) 
        c7 = self.dec1(u1) 

        u2 = self.up2(c7)
        u2 = torch.cat([u2, c4], dim=1) 
        c8 = self.dec2(u2) 

        u3 = self.up3(c8)
        u3 = torch.cat([u3, c3], dim=1) 
        c9 = self.dec3(u3) 

        u4 = self.up4(c9)
        u4 = torch.cat([u4, c2], dim=1)
        c10 = self.dec4(u4)

        u5 = self.up5(c10)
        u5 = torch.cat([u5, c1], dim=1)
        c11 = self.dec5(u5)

        outputs = self.final_conv(c11) 
        outputs = outputs.reshape(-1, 4, 2, 512, 512)
        return outputs
    

class NewUNet3(nn.Module): # Final version of model
    def __init__(self, size=(2,512,512)):
        super(NewUNet3, self).__init__()
        self.n_channels = size[0]
        self.n_bins = size[1]
        self.n_frames = size[2]

        def conv_block(in_channels, out_channels, dropout=0.0, num_groups=8):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            )
        
        # (-, 2, 512, 512)
        self.enc1 = conv_block(self.n_channels, 64) # (-, 64, 512, 512)
        self.pool1 = nn.MaxPool2d(2) # (-, 64, 256, 256)

        self.enc2 = conv_block(64, 128) # (-, 128, 256, 256)
        self.pool2 = nn.MaxPool2d(2) # (-, 128, 128, 128)

        self.enc3 = conv_block(128, 256) # (-, 256, 128, 128)
        self.pool3 = nn.MaxPool2d(2) # (-, 256, 64, 64)

        self.enc4 = conv_block(256, 512, 0.05) # (-, 512, 64, 64)
        self.pool4 = nn.MaxPool2d(2) # (-, 512, 32, 32)

        self.enc5 = conv_block(512, 1024, 0.1) # (-, 1024, 32, 32)
        self.pool5 = nn.MaxPool2d(2) # (-, 1024, 16, 16)

        self.bottleneck = conv_block(1024, 2048, 0.15) # (-, 2048, 16, 16)

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2) # (-, 2048, 32, 32)
        self.dec1 = conv_block(2048, 1024, 0.1) # (-, 1024, 32, 32)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # (-, 1024, 64, 64)
        self.dec2 = conv_block(1024, 512, 0.05) # (-, 512, 64, 64)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # (-, 512, 128, 128)
        self.dec3 = conv_block(512, 256) # (-, 256, 128, 128)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # (-, 256, 256, 256)
        self.dec4 = conv_block(256, 128) # (-, 128, 512, 256)

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # (-, 128, 512, 512)
        self.dec5 = conv_block(128, 64) # (-, 64, 512, 512)

        self.final_conv = nn.Conv2d(64, 4*self.n_channels, kernel_size=1) # (-, 8, 512, 512)

    def forward(self, x):
        c1 = self.enc1(x) 
        p1 = self.pool1(c1) 

        c2 = self.enc2(p1) 
        p2 = self.pool2(c2) 

        c3 = self.enc3(p2) 
        p3 = self.pool3(c3) 

        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        c5 = self.enc5(p4)
        p5 = self.pool5(c5)

        c6 = self.bottleneck(p5) 

        u1 = self.up1(c6)
        u1 = torch.cat([u1, c5], dim=1) 
        c7 = self.dec1(u1) 

        u2 = self.up2(c7)
        u2 = torch.cat([u2, c4], dim=1) 
        c8 = self.dec2(u2) 

        u3 = self.up3(c8)
        u3 = torch.cat([u3, c3], dim=1) 
        c9 = self.dec3(u3) 

        u4 = self.up4(c9)
        u4 = torch.cat([u4, c2], dim=1)
        c10 = self.dec4(u4)

        u5 = self.up5(c10)
        u5 = torch.cat([u5, c1], dim=1)
        c11 = self.dec5(u5)

        outputs = self.final_conv(c11) 
        outputs = outputs.reshape(-1, 4, self.n_channels, self.n_bins, self.n_frames)

        return outputs