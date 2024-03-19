import sys
import os
import logging
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/decoder.log",
)


class Decoder(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = self.decoder_block()

    def decoder_block(self):
        layers = OrderedDict()
        layers["deconv1"] = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=2,
            stride=2,
        )
        return nn.Sequential(layers)

    def forward(self, x, skip_info):
        if x is not None and skip_info is not None:
            return torch.cat((self.model(x), skip_info), dim=1)
        else:
            raise ValueError("Input and skip_info cannot be None".capitalize())


if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(
        description="Define the Decoder block for U-Net".title()
    )
    parser.add_argument(
        "--decoder", action="store_true", help="Decoder block".capitalize()
    )
    args = parser.parse_args()

    if args.decoder:
        decoder = Decoder(in_channels=1024, out_channels=512)
        logging.info(decoder)
    else:
        raise ValueError("Define the arguments for the decoder".capitalize())
