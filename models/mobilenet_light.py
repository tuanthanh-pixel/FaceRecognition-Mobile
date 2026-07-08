import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.block = nn.Sequential(

            # Depthwise
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # Pointwise
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MobileNetLight(nn.Module):

    def __init__(self, embedding_size=512):
        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(3, 32, 2),

            DepthwiseSeparableConv(32, 64, 1),

            DepthwiseSeparableConv(64, 128, 2),

            DepthwiseSeparableConv(128, 128, 1),

            DepthwiseSeparableConv(128, 256, 2),

            DepthwiseSeparableConv(256, 256, 1),

            DepthwiseSeparableConv(256, 512, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Không dùng bias vì sau đó có BatchNorm1d
        self.embedding = nn.Linear(
            512,
            embedding_size,
            bias=False
        )

        # Thêm BatchNorm1d
        self.bn = nn.BatchNorm1d(
            embedding_size
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.embedding(x)

        # Chuẩn hóa embedding
        x = self.bn(x)

        # L2 Normalize
        x = F.normalize(
            x,
            p=2,
            dim=1
        )

        return x