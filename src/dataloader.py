import sys
import os
import logging
import argparse
import zipfile
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

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
    """
    A loader class for preparing and loading datasets for U-Net model training.

    | Attributes   | Type          | Description                                           |
    |--------------|---------------|-------------------------------------------------------|
    | image_path   | str           | Path to the image folder.                             |
    | batch_size   | int           | Batch size for the DataLoader.                        |
    | directory    | str or None   | The directory path of the unzipped images.            |
    | categories   | list or None  | A list of category names in the dataset.              |
    | base_images  | list          | A list of transformed base images.                    |
    | mask_images  | list          | A list of transformed mask images.                    |
    | is_mask      | str           | Identifier for mask images.                           |

    | Parameters   | Type  | Description                                        |
    |--------------|-------|----------------------------------------------------|
    | image_path   | str   | Path to the zip file containing the dataset images.|
    | batch_size   | int   | The number of images to load in each batch.        |

    Examples:
        # Example of initializing the Loader with a specific image path and batch size.
        loader = Loader(image_path='/path/to/your/dataset.zip', batch_size=32)

        # Example of unzipping the dataset folder.
        loader.unzip_folder()

        # Example of creating a DataLoader for your dataset.
        # This assumes you've implemented and called necessary methods
        # to process your images and prepare them accordingly.
        dataloader = loader.create_dataloader()

        # Now `dataloader` can be used in your model training loop.
    """

    def __init__(self, image_path=None, batch_size=32):
        self.image_path = image_path
        self.batch_size = batch_size
        self.directory = None
        self.categories = None
        self.base_images = list()
        self.mask_images = list()
        self.is_mask = "mask"

    def base_transformation(self):
        """
        Defines the transformation pipeline for mask images.

        | Returns                  | Description                                             |
        |--------------------------|---------------------------------------------------------|
        | torchvision.transforms.Compose | A composition of image transformations for masks. |
        """
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
        """
        Unzips the dataset folder and prepares the directory for image loading.

        This method checks if the RAW_PATH directory exists, cleans it if necessary, and then
        unzips the provided dataset into this directory.
        """
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
        """
        Creates a transformed image from a given path and type (base or mask).

        | Parameters   | Type  | Description                                      |
        |--------------|-------|--------------------------------------------------|
        | folder_path  | str   | Path to the folder containing the image.        |
        | image        | str   | Name of the image file (for base images) or mask image file. |
        | type         | str   | Specifies whether the image is a 'base' image or a 'mask'.   |
        """
        if os.path.exists(RAW_PATH):
            clean(RAW_PATH)
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(RAW_PATH))
        else:
            os.makedirs(RAW_PATH)

    def create_image_from_path(self, **kwargs):
        """
        Creates and returns a transformed image from a given path, based on the image type (base or mask).

        | Parameters   | Type  | Description                                      |
        |--------------|-------|--------------------------------------------------|
        | folder_path  | str   | Path to the folder containing the image.        |
        | image        | str   | Name of the base image file. Only required for base images. |
        | mask_image   | str   | Name of the mask image file. Only required for mask images. |
        | type         | str   | Specifies whether the image is a 'base' image or a 'mask'.   |

        | Returns      | Description                                             |
        |--------------|---------------------------------------------------------|
        | PIL.Image or None | The transformed image as a PIL Image object, or None if an error occurs. |
        """
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
        """
        Creates a DataLoader from the processed and transformed images.

        This method organizes the base and mask images into pairs, prepares them for
        training by applying the necessary transformations, and then bundles them into
        a DataLoader object.

        | Returns                  | Description                                       |
        |--------------------------|---------------------------------------------------|
        | torch.utils.data.DataLoader | DataLoader containing paired and transformed images. |

        | Raises                   | Description                                       |
        |--------------------------|---------------------------------------------------|
        | Exception                | If the PROCESSED_PATH does not exist.             |
        """
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
