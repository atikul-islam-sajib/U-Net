import sys
import os
import unittest
import torch

sys.path.append("src/")

from utils import load_pickle
from config import PROCESSED_PATH
from dataloader import Loader
from encoder import Encoder
from decoder import Decoder


class UnitTest(unittest.TestCase):
"""
    Unit Tests for U-Net Components

    This script contains unit tests for verifying the functionality of the U-Net architecture components, including data loaders, encoder blocks, and decoder blocks. It ensures the integrity and correctness of data processing, model input/output dimensions, and the integration between different parts of the U-Net model. The tests cover the following:

    - Data loading and preprocessing: Validates the data loader's ability to correctly load and preprocess the dataset.
    - Input data integrity: Confirms that input images to the U-Net have the expected dimensions and channels (RGB and Grayscale checks).
    - Model functionality: Tests the encoder and decoder blocks for correct tensor shape transformations, ensuring proper forward propagation through the network components.
    - Skip connections: Verifies the decoder's ability to concatenate skip connection information correctly with its input.

    Dependencies:
    - Python Standard Library: os, sys, unittest
    - Third-party Libraries: torch
    - Local Scripts: utils (for `load_pickle`), config (for `PROCESSED_PATH`), dataloader (for `Loader`), encoder, decoder

    Structure:
    - setUp: Initializes the test environment by loading necessary data and models.
    - test_quantity_data: Checks if the total number of data points loaded matches the expected quantity.
    - test_RGB: Ensures that RGB input images have the correct dimensions.
    - test_GRAY: Validates that Grayscale label images have the expected dimensions.
    - test_total_data: Verifies the total number of data and label points combined matches the expected count.
    - test_encoder_block: Confirms that the encoder block outputs tensors of the correct shape.
    - test_decoder_block: Assesses whether the decoder block correctly handles inputs and skip connections, outputting tensors of the expected shape.

    To run these tests, execute the script directly from the command line. Ensure that all dependencies are installed and that the necessary data is available in the specified `PROCESSED_PATH`.
    """

    def setUp(self):
        self.dataloader = load_pickle(os.path.join(PROCESSED_PATH, "dataloader.pkl"))
        self.encoder = Encoder(in_channels=3, out_channels=64)
        self.decoder = Decoder(in_channels=64, out_channels=64)
        self.noise_samples = torch.randn(64, 3, 256, 256)
        self.skip_info = torch.rand(64, 64, 256, 256)

    def test_quantity_data(self):
        self.assertEqual(sum(data.size(0) for data, label in self.dataloader), 780)

    def test_RGB(self):
        data, _ = next(iter(self.dataloader))
        self.assertEqual(data.size(), torch.Size([32, 3, 256, 256]))

    def test_GRAY(self):
        _, label = next(iter(self.dataloader))
        self.assertEqual(label.size(), torch.Size([32, 1, 256, 256]))

    def test_total_data(self):
        self.assertEqual(
            sum(data.size(0) + label.size(0) for data, label in self.dataloader),
            780 * 2,
        )

    def test_encoder_block(self):
        self.assertEqual(
            self.encoder(self.noise_samples).shape, torch.Size([64, 64, 256, 256])
        )

    def test_decoder_block(self):
        self.assertEqual(
            self.decoder(self.encoder(self.noise_samples), self.skip_info),
            torch.Size([64, 128, 256, 256]),
        )


if __name__ == "__main__":
    unittest.main()
