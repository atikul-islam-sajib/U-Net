# import sys
# import os
# import unittest

# sys.path.append("src/")

# from utils import load_pickle
# from config import PROCESSED_PATH
# from dataloader import Loader


# class UnitTest(unittest.TestCase):
#     def setUp(self):
#         self.dataloader = load_pickle(os.path.join(PROCESSED_PATH, "dataloader.pkl"))

#     def test_quantity_data(self):
#         self.assertEqual(sum(data.size(0) for data, label in self.dataloader), 780)


# if __name__ == "__main__":
#     unittest.main()

import joblib

data = joblib.load(
    "/Users/shahmuhammadraditrahman/Desktop/U-Net/data/processed/dataloader.pkl"
)

data, label = next(iter(data))

print(data.shape)
