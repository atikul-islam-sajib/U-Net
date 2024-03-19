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
from UNet import UNet


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
        """
        Set up the testing environment before running each test.

        Loads the dataloader from a pickle file, initializes the Encoder and Decoder with specified channel sizes, and generates sample noise and skip information tensors for testing model inputs.
        """
        # self.dataloader = load_pickle(os.path.join(PROCESSED_PATH, "dataloader.pkl"))
        self.encoder = Encoder(in_channels=3, out_channels=64)
        self.decoder = Decoder(in_channels=64, out_channels=64)
        self.UNet = UNet()
        self.noise_samples = torch.randn(64, 3, 256, 256)
        self.skip_info = torch.rand(64, 64, 256, 256)
        self.noise_image = torch.randn(64, 3, 256, 256)

    def test_quantity_data(self):
        """
        Test if the total number of data points in the dataloader matches the expected quantity.
        """
        self.assertEqual(sum(data.size(0) for data, label in self.dataloader), 780)

    def test_RGB(self):
        """
        Ensure that the input images are in the RGB format with the correct dimensions.
        """
        data, _ = next(iter(self.dataloader))
        self.assertEqual(data.size(), torch.Size([32, 3, 256, 256]))

    def test_GRAY(self):
        """
        Verify that the label images are in the Grayscale format with the expected dimensions.
        """
        _, label = next(iter(self.dataloader))
        self.assertEqual(label.size(), torch.Size([32, 1, 256, 256]))

    def test_total_data(self):
        """
        Check if the combined count of data and label points for all items in the dataloader matches the expected total.
        """
        self.assertEqual(
            sum(data.size(0) + label.size(0) for data, label in self.dataloader),
            780 * 2,
        )

    def test_encoder_block(self):
        """
        Validate that the encoder block processes input noise samples and outputs tensors of the correct shape.
        """
        self.assertEqual(
            self.encoder(self.noise_samples).shape, torch.Size([64, 64, 256, 256])
        )

    def test_decoder_block(self):
        """
        Assess the decoder block's ability to process encoded inputs and skip information, verifying the output shape.

        Note: This test is expected to fail based on the provided implementation. The correct assertion should compare against a torch.Size, not a direct torch.Size object.
        """
        self.assertEqual(
            self.decoder(self.encoder(self.noise_samples), self.skip_info),
            torch.Size([64, 128, 256, 256]),
        )

    def test_unet_shape(self):
        """
        Tests if the UNet model produces the expected output shape.

        This test verifies that the output of the UNet model, when given a predefined noise image, matches the expected tensor shape. The expected shape is specified to ensure the model's output dimensions align with the input dimensions for typical use cases, demonstrating the model's capability to maintain input size through the network.

        ### Assertions

        - Asserts that the shape of the UNet model's output matches the expected shape `[64, 1, 256, 256]`, where 64 is the batch size, 1 is the number of output channels, and 256x256 is the spatial dimension of the output tensor.
        """
        self.assertEqual(
            self.UNet(self.noise_image).shape, torch.Size([64, 1, 256, 256])
        )

    def test_total_params_unet(self):
        """
        Tests the total number of parameters in the UNet model.

        This test calculates the total number of trainable parameters in the UNet model to verify it matches the expected number. Ensuring the correct number of parameters is crucial for validating the model's size and complexity, which impacts both performance and resource requirements.

        ### Assertions

        - Asserts that the total number of trainable parameters in the UNet model is exactly 31,037,633. This total parameter count is an important characteristic of the model, indicating its complexity and capacity to learn from data.
        """
        self.assertEqual(
            sum(params.numel() for params in self.UNet.parameters()), 31037633
        )


if __name__ == "__main__":
    unittest.main()
