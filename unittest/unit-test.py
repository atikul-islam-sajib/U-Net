import sys
import os
import unittest

sys.path.append("src/")

from config import PROCESSED_PATH
from dataloader import Loader


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.dataloader = os.path.join(PROCESSED_PATH, "dataloader.pkl")

    def test_quantity_data(self):
        self.assertEqual(sum(data.size(0) for data, label in self.dataloader), 780)


if __name__ == "__main__":
    unittest.main()
