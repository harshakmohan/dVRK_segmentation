import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class UNetV2(torch.nn.Module):

    def __init__(self, nchannel=3, nclass=1):
        super().__init__()
        self.iput = self.conv(nchannel, 64)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.enc1 = self.conv(64, 128)
        self.enc2 = self.conv(128, 256)
        self.enc3 = self.conv(256, 512)
        self.enc4 = self.conv(512, 1024 // 2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv(1024, 512 // 2)
        self.dec2 = self.conv(512, 256 // 2)
        self.dec3 = self.conv(256, 128 // 2)
        self.dec4 = self.conv(128, 64)
        self.oput = torch.nn.Conv2d(64, nclass, kernel_size=1)

    def forward(self, x):
        x1 = self.iput(x)  # input
        # encoder layers
        x2 = self.maxpool(x1)
        x2 = self.enc1(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc2(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc3(x4)
        x5 = self.maxpool(x4)
        x5 = self.enc4(x5)
        # decoder layers with skip connections and attention gates
        x = self.upsample(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec4(x)
        return self.oput(x)  # output

    def conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True))

