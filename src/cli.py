import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/cli.log",
)

sys.path.append("src/")

from dataloader import Loader
from UNet import UNet
from trainer import Trainer
from test import Charts


def cli():
    parser = argparse.ArgumentParser(
        description="Define the CLI coding for U-Net".title()
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the image folder".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--smooth_value",
        type=float,
        default=0.01,
        help="Define the smooth value".capitalize(),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Define the number of epochs".capitalize(),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0002,
        help="Define the learning rate".capitalize(),
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Define the beta1 value".capitalize()
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Define the beta2 value".capitalize()
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Define the device".capitalize()
    )
    parser.add_argument(
        "--l2",
        type=bool,
        default=False,
        help="Define if L2 regularization is applied".capitalize(),
    )
    parser.add_argument(
        "--criterion",
        type=bool,
        default=False,
        help="Define if criterion is applied".capitalize(),
    )
    parser.add_argument(
        "--display", type=bool, default=True, help="Display the progress".capitalize()
    )
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
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:
        if (
            args.image_path
            and args.batch_size
            and args.smooth_value
            and args.epochs
            and args.learning_rate
            and args.beta1
            and args.beta2
            and args.device
            and args.display
        ):
            logging.info("Creating dataloader".capitalize())

            loader = Loader(image_path=args.image_path, batch_size=args.batch_size)
            loader.unzip_folder()
            _ = loader.create_dataloader()

            logging.info("Dataloader created".capitalize())

            trainer = Trainer(
                smooth_value=args.smooth_value,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                beta1=args.beta1,
                beta2=args.beta2,
                device=args.device,
                use_l2=args.l2,
                use_criterion=args.criterion,
            )

            trainer.train()

            logging.info("Model trained".capitalize())

    else:
        if args.samples and args.device:
            logging.info("Testing the model".capitalize())

            test = Charts(samples=args.samples, device=args.device)
            test.test()
            Charts.plot_loss_curves()

            logging.info("Model tested".capitalize())


if __name__ == "__main__":
    cli()
