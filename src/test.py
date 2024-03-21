import sys
import os
import logging
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/test.log",
)


sys.path.append("src/")

from config import PROCESSED_PATH, IMAGES_PATH, TEST_IMAGE_PATH, BEST_MODEL_PATH
from utils import load_pickle, device_init
from UNet import UNet


class Charts:
    def __init__(self, samples=4, device="mps"):
        self.samples = samples
        self.device = device_init(device=device)
        self.total_images = list()
        self.total_masks = list()

    def select_best_model(self):
        if os.path.exists(BEST_MODEL_PATH):
            model = torch.load(os.path.join(BEST_MODEL_PATH, "last_model.pth"))
            return model
        else:
            raise Exception(
                "No best model found. Please run train.py first.".capitalize()
            )

    def plot_data_comparison(self, **kwargs):
        model = self.select_best_model()
        images = model(kwargs["images"].to(self.device))
        plt.figure(figsize=(20, 15))

        for index, image in enumerate(images):
            plt.subplot(3 * 4, 3 * 5, 3 * index + 1)
            temp_base_image = (
                kwargs["images"][index].permute(1, 2, 0).cpu().detach().numpy()
            )
            temp_base_image = (temp_base_image - temp_base_image.min()) / (
                temp_base_image.max() - temp_base_image.min()
            )
            plt.imshow(temp_base_image, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(3 * 4, 3 * 5, 3 * index + 2)
            mask = kwargs["masks"][index].permute(1, 2, 0).cpu().detach().numpy()
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            plt.imshow(mask, cmap="gray")
            plt.title("Real")
            plt.axis("off")

            plt.subplot(3 * 4, 3 * 5, 3 * index + 3)
            image = image.permute(1, 2, 0).cpu().detach().numpy()
            image = (image - image.min()) / (image.max() - image.min())
            plt.imshow(image, cmap="gray")
            plt.title("Generated")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def test(self):
        if os.path.exists(PROCESSED_PATH):
            dataloader = load_pickle(
                os.path.join(PROCESSED_PATH, "train_dataloader.pkl")
            )
            for _ in range(self.samples):
                images, masks = next(iter(dataloader))
                if len(self.total_images) != self.samples:
                    self.total_images.append(images)
                    self.total_masks.append(masks)

            try:
                self.plot_data_comparison(
                    images=self.total_images, masks=self.total_masks
                )
            except Exception as e:
                logging.exception(
                    "The exception caught in the section - {}".format(e).capitalize()
                )
        else:
            raise Exception(
                "No processed data found. Please run preprocess.py first.".capitalize()
            )


if __name__ == "__main__":
    test = Charts(samples=20, device="mps")
    print(test.test())
