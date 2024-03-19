import sys
import os
import logging
import argparse
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/U-Net.log",
)

sys.path.append("src/")

from encoder import Encoder
from decoder import Decoder


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder_layer1 = Encoder(in_channels=3, out_channels=64)
        self.encoder_layer2 = Encoder(in_channels=64, out_channels=128)
        self.encoder_layer3 = Encoder(in_channels=128, out_channels=256)
        self.encoder_layer4 = Encoder(in_channels=256, out_channels=512)
        self.bottom_layer = Encoder(in_channels=512, out_channels=1024)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.intermediate_layer1 = Encoder(in_channels=1024, out_channels=512)
        self.intermediate_layer2 = Encoder(in_channels=512, out_channels=256)
        self.intermediate_layer3 = Encoder(in_channels=256, out_channels=128)
        self.intermediate_layer4 = Encoder(in_channels=128, out_channels=64)

        self.decoder_layer1 = Decoder(in_channels=1024, out_channels=512)
        self.decoder_layer2 = Decoder(in_channels=512, out_channels=256)
        self.decoder_layer3 = Decoder(in_channels=256, out_channels=128)
        self.decoder_layer4 = Decoder(in_channels=128, out_channels=64)

        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder path
        enc1_out = self.encoder_layer1(x)
        pooled_enc1 = self.max_pool(enc1_out)

        enc2_out = self.encoder_layer2(pooled_enc1)
        pooled_enc2 = self.max_pool(enc2_out)

        enc3_out = self.encoder_layer3(pooled_enc2)
        pooled_enc3 = self.max_pool(enc3_out)

        enc4_out = self.encoder_layer4(pooled_enc3)
        pooled_enc4 = self.max_pool(enc4_out)

        bottom_out = self.bottom_layer(pooled_enc4)

        # Decoder path
        dec1_input = self.decoder_layer1(bottom_out, enc4_out)
        dec1_out = self.intermediate_layer1(dec1_input)

        dec2_input = self.decoder_layer2(dec1_out, enc3_out)
        dec2_out = self.intermediate_layer2(dec2_input)

        dec3_input = self.decoder_layer3(dec2_out, enc2_out)
        dec3_out = self.intermediate_layer3(dec3_input)

        dec4_input = self.decoder_layer4(dec3_out, enc1_out)
        dec4_out = self.intermediate_layer4(dec4_input)

        # Final output
        final_output = self.final_layer(dec4_out)

        return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet model".title())
    parser.add_argument(
        "--unet", action="store_true", help="Run UNet model".capitalize()
    )
    args = parser.parse_args()

    if args.unet:
        model = UNet()
        logging.info(model)
    else:
        raise ValueError("Use the appropriate flag to run the model".capitalize())
