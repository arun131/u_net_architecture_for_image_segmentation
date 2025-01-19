import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def up(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = down(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = down(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = down(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = down(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = down(512, 1024)

        self.up1 = up(1024, 512)
        self.conv6 = down(1024, 512)
        self.up2 = up(512, 256)
        self.conv7 = down(512, 256)
        self.up3 = up(256, 128)
        self.conv8 = down(256, 128)
        self.up4 = up(128, 64)
        self.conv9 = down(128, 64)

        self.conv10 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        u1 = self.up1(c5)
        cat1 = torch.cat([u1, c4], dim=1)
        c6 = self.conv6(cat1)
        u2 = self.up2(c6)
        cat2 = torch.cat([u2, c3], dim=1)
        c7 = self.conv7(cat2)
        u3 = self.up3(c7)
        cat3 = torch.cat([u3, c2], dim=1)
        c8 = self.conv8(cat3)
        u4 = self.up4(c8)
        cat4 = torch.cat([u4, c1], dim=1)
        c9 = self.conv9(cat4)

        output = self.conv10(c9)

        return output

# Example usage
# Create an instance of the UNet model
model = UNet(in_channels=3, out_channels=1)