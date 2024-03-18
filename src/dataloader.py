import sys
import logging
import argparse
import os
import zipfile
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/dataloader.log",
)

sys.path.append("src/")

from config import RAW_PATH, PROCESSED_PATH
from utils import load_pickle, config, clean


class Loader:
    def __init__(self, image_path=None, batch_size=32):
        self.image_path = image_path
        self.batch_size = batch_size
        self.directory = None
        self.categories = None
        self.base_images = list()
        self.mask_images = list()
        self.is_mask = "mask"

    def base_transformation(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    (config()["data"]["image_width"], config()["data"]["image_height"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        config()["data"]["transforms"],
                        config()["data"]["transforms"],
                        config()["data"]["transforms"],
                    ],
                    std=[
                        config()["data"]["transforms"],
                        config()["data"]["transforms"],
                        config()["data"]["transforms"],
                    ],
                ),
            ]
        )

    def mask_transformation(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    (config()["data"]["image_width"], config()["data"]["image_height"])
                ),
                transforms.ToTensor(),
                transforms.Grayscale(
                    num_output_channels=config()["data"]["gray_channels"]
                ),
                transforms.Normalize(
                    mean=[
                        config()["data"]["transforms"],
                    ],
                    std=[
                        config()["data"]["transforms"],
                    ],
                ),
            ]
        )

    def unzip_folder(self):
        if os.path.exists(RAW_PATH):
            clean(RAW_PATH)
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(RAW_PATH))
        else:
            os.makedirs(RAW_PATH)

    def create_image_from_path(self, **kwargs):
        if kwargs["type"] == "base":
            self.base_transformation()(
                Image.fromarray(
                    cv2.imread(os.path.join(kwargs["folder_path"], kwargs["image"]))
                )
            )
        else:
            self.mask_transformation()(
                Image.fromarray(
                    cv2.imread(
                        os.path.join(kwargs["folder_path"], kwargs["mask_image"]),
                        cv2.IMREAD_GRAYSCALE,
                    )
                )
            )

    def create_dataloader(self):
        self.directory = os.path.join(RAW_PATH, os.listdir(RAW_PATH)[0])
        self.categories = os.listdir(self.directory)

        for category in self.categories:
            folder_path = os.path.join(self.directory, category)
            for image in os.listdir(folder_path):
                if self.is_mask in image:
                    continue

                base_image = image.split(".")[0]
                extension = image.split(".")[1]
                mask_image = "{}.{}".format(base_image, extension)

                self.base_images.append(
                    self.create_image_from_path(
                        folder_path=folder_path, image=image, type="base"
                    )
                )
                self.mask_images.append(
                    self.create_image_from_path(
                        folder_path=folder_path, mask_image=mask_image, type="mask"
                    )
                )

        if os.path.exists(PROCESSED_PATH):
            dataloader = DataLoader(
                dataset=list(zip(self.base_images, self.mask_images)),
                batch_size=self.batch_size,
                shuffle=True,
            )
            load_pickle(
                value=self.base_images,
                filename=os.path.join(PROCESSED_PATH, "base_images.pkl"),
            )
            load_pickle(
                value=self.mask_images,
                filename=os.path.join(PROCESSED_PATH, "mask_images.pkl"),
            )
            load_pickle(
                value=dataloader,
                filename=os.path.join(PROCESSED_PATH, "dataloader.pkl"),
            )

            return dataloader

        else:
            raise Exception("PROCESSED_PATH does not exist".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the dataloader for U-Net".title()
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the image folder".capitalize(),
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for the dataloader".capitalize(),
        required=True,
    )
    args = parser.parse_args()

    if args.image_path and args.batch_size:
        logging.info("Creating dataloader".capitalize())

        loader = Loader(image_path=args.image_path, batch_size=args.batch_size)
        loader.unzip_folder()
        dataloader = loader.create_dataloader()

        logging.info("Dataloader created".capitalize())
    else:
        raise ValueError("Arguments need to be defined properly".capitalize())
