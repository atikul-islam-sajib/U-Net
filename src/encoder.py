import sys
import logging
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/encoder.log",
)


class Encoder(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = self.encoder_block()

    def encoder_block(self):
        layers = OrderedDict()
        layers["conv1"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        layers["relu1"] = nn.ReLU(inplace=True)
        layers["conv2"] = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        layers["batch_norm1"] = nn.BatchNorm2d(self.out_channels)
        layers["relu2"] = nn.ReLU(inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        return self.model(x) if x is not None else None


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(
        description="Define the Encoder block for U-Net".title()
    )
    parser.add_argument(
        "--encoder", action="store_true", help="Encoder block".capitalize()
    )
    args = parser.parse_args()
    if args.encoder:
        encoder = Encoder(in_channels=3, out_channels=64)
        noise_samples = torch.randn(64, 3, 256, 256)
        print(encoder(noise_samples).shape)
    else:
        raise ValueError("Define the arguments in an appropriate way".capitalize())
