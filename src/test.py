import sys
import os
import logging
import argparse
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/test.log",
)


sys.path.append("src/")

from config import (
    PROCESSED_PATH,
    IMAGES_PATH,
    TEST_IMAGE_PATH,
    BEST_MODEL_PATH,
    GIF_PATH,
)
from utils import load_pickle, device_init
from UNet import UNet


class Charts:
    def __init__(self, samples=4, device="mps"):
        self.samples = samples
        self.device = device_init(device=device)

    def select_best_model(self):
        if os.path.exists(BEST_MODEL_PATH):
            model = torch.load(os.path.join(BEST_MODEL_PATH, "last_model.pth"))
            return model
        else:
            raise Exception(
                "No best model found. Please run train.py first.".capitalize()
            )

    def plot_data_comparison(self, **kwargs):
        model = UNet().to(self.device)
        model.load_state_dict(self.select_best_model())
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

        try:
            plt.tight_layout()
            if os.path.exists(TEST_IMAGE_PATH):
                plt.savefig(os.path.join(TEST_IMAGE_PATH, "result.png"))
        except Exception as e:
            logging.exception(
                "The exception caught in the section - {}".format(e).capitalize()
            )
        finally:
            plt.show()

    def generate_gif(self):
        if os.path.exists(GIF_PATH) and os.path.exists(IMAGES_PATH):
            images = [
                imageio.imread(os.path.join(IMAGES_PATH, image))
                for image in os.listdir(IMAGES_PATH)
            ]
            imageio.mimsave(os.path.join(GIF_PATH, "result.gif"), images, "GIF")
        else:
            raise Exception("No gif found. Please run train.py first.".capitalize())

    def test(self):
        if os.path.exists(PROCESSED_PATH):
            dataloader = load_pickle(os.path.join(PROCESSED_PATH, "dataloader.pkl"))
            images, masks = next(iter(dataloader))
            images = images.to(self.device)
            masks = masks.to(self.device)
            images = images[0 : self.samples]
            masks = masks[0 : self.samples]
            try:
                self.plot_data_comparison(images=images, masks=masks)
                self.generate_gif()
            except Exception as e:
                logging.exception(
                    "The exception caught in the section - {}".format(e).capitalize()
                )
        else:
            raise Exception(
                "No processed data found. Please run preprocess.py first.".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model".title())
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        choices=[
            10,
            20,
        ],
        help="Number of samples to plot".capitalize(),
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to use".capitalize()
    )

    args = parser.parse_args()

    if args.samples and args.device:
        test = Charts(samples=args.samples, device=args.device)
        test.test()
    else:
        raise Exception(
            "Please provide the number of samples and the device to use.".capitalize()
        )
