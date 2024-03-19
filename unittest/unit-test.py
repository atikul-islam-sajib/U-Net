import sys
import os
import unittest
import torch

sys.path.append("src/")

from utils import load_pickle
from config import PROCESSED_PATH
from dataloader import Loader
from encoder import Encoder


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.dataloader = load_pickle(os.path.join(PROCESSED_PATH, "dataloader.pkl"))
        self.encoder = Encoder(in_channels=3, out_channels=64)
        self.noise_samples = torch.randn(64, 3, 256, 256)

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


if __name__ == "__main__":
    unittest.main()
