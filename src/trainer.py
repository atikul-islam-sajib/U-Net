import sys
import os
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m-%d %H:%M",
    filemode="w",
    filename="./logs/train.log",
)

sys.path.append("src/")

from config import PROCESSED_PATH, CHECKPOINTS_PATH, IMAGES_PATH, BEST_MODEL_PATH
from utils import device_init, weight_init, config, load_pickle
from UNet import UNet


class Trainer:
    """
    A class to encapsulate the training process for the U-Net model on a specified dataset.

    | Attributes           | Description |
    |----------------------|-------------|
    | epochs               | Total number of epochs to train the model. |
    | smooth_value         | Smoothing value used in dice loss calculation to prevent division by zero. |
    | learning_rate        | Learning rate for the Adam optimizer. |
    | beta1                | Beta1 value for the Adam optimizer. |
    | beta2                | Beta2 value for the Adam optimizer. |
    | device               | Device type ('cuda', 'cpu', 'mps') on which the model will be trained. |
    | display              | Flag to control the verbosity of the training process output. |
    | is_l2                | Flag indicating whether L2 regularization is applied. |
    | is_criterion         | Flag indicating whether an additional criterion is applied. |
    | history              | Dictionary to store training and validation loss history. |

    | Parameters       | Type    | Default | Description |
    |------------------|---------|---------|-------------|
    | smooth_value     | float   | 0.01    | Smoothing value for dice loss calculation. |
    | epochs           | int     | 100     | Total number of training epochs. |
    | learning_rate    | float   | 0.0002  | Learning rate for optimizer. |
    | beta1            | float   | 0.5     | Beta1 value for Adam optimizer. |
    | beta2            | float   | 0.999   | Beta2 value for Adam optimizer. |
    | device           | str     | 'mps'   | Training device. |
    | display          | bool    | True    | Whether to display training progress. |
    | use_l2           | bool    | False   | Apply L2 regularization if True. |
    | use_criterion    | bool    | False   | Apply additional criterion if True. |

    Methods:
        __setup__: Configures the training environment by initializing the model, optimizer, and data loaders.
        l2_regularization: Calculates L2 norm of model parameters.
        dice_loss: Computes the dice loss between the predicted masks and the true masks.
        train_model: Conducts a single training step, including forward and backward passes.
        val_model: Validates the model on the validation dataset.
        save_checkpoints: Saves model checkpoints during training.
        show_progress: Prints or logs the training progress.
        train: Main training loop that iterates over epochs and batches.
    """

    def __init__(
        self,
        smooth_value=0.01,
        epochs=100,
        learning_rate=0.0002,
        beta1=0.5,
        beta2=0.999,
        device="mps",
        display=True,
        use_l2=False,
        use_criterion=False,
    ):
        self.epochs = epochs
        self.smooth_value = smooth_value
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = device
        self.display = display
        self.is_l2 = use_l2
        self.is_criterion = use_criterion
        self.history = {"train_loss": list(), "val_loss": list()}

    def __setup__(self):
        try:
            self.model = UNet()
        except Exception as e:
            raise Exception("U-Net model could not be loaded".capitalize())
        else:
            self.device = device_init(device=self.device)
            self.model = self.model.apply(weight_init)
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
            )
            self.criterion = nn.BCELoss()
        finally:
            if os.path.exists(PROCESSED_PATH):
                self.train_dataloader = load_pickle(
                    os.path.join(PROCESSED_PATH, "train_dataloader.pkl")
                )
                self.val_dataloader = load_pickle(
                    os.path.join(PROCESSED_PATH, "val_dataloader.pkl")
                )
            else:
                raise Exception("Processed path cannot be found".capitalize())

    def l2_regularization(self, model):
        """
        Calculates the L2 regularization term for the model parameters.

        Parameters:
            model (torch.nn.Module): The neural network model whose parameters are considered for L2 regularization.

        Returns:
            torch.Tensor: The L2 norm of the model parameters.
        """
        return sum(torch.norm(params, p=2) for params in model.parameters())

    def dice_loss(self, predicted, target):
        """
        Calculates the Dice loss between the predicted and target masks, facilitating the training of the model
        for segmentation tasks.

        Parameters:
            predicted (torch.Tensor): The predicted masks by the model.
            target (torch.Tensor): The ground truth masks.

        Returns:
            torch.Tensor: The computed Dice loss value.
        """
        predicted = predicted.view(-1)
        target = target.view(-1)
        intersection = (predicted * target).sum() + self.smooth_value
        return (2.0 * intersection) / (
            predicted.sum() + target.sum() + self.smooth_value
        )

    def train_model(self, **kwargs):
        """
        Executes a single training step including the forward pass, loss calculation, and the backward pass.

        Parameters:
            **kwargs: Arbitrary keyword arguments including:
                - train_image (torch.Tensor): The batch of training images.
                - train_mask (torch.Tensor): The batch of ground truth masks for training images.

        Returns:
            float: The training loss for the current step.
        """
        self.optimizer.zero_grad()

        predicted_mask = self.model(kwargs["train_image"])
        if self.is_l2 == True:
            train_loss = self.dice_loss(
                predicted_mask, kwargs["train_mask"]
            ) + config()["model"]["lambda"] * self.l2_regularization(model=self.model)
        elif self.is_criterion == True:
            train_loss = self.dice_loss(
                predicted_mask, kwargs["train_mask"]
            ) + self.criterion(predicted_mask, kwargs["train_mask"])
        else:
            train_loss = self.dice_loss(predicted_mask, kwargs["train_mask"])

        train_loss.backward(retain_graph=True)
        self.optimizer.step()

        return train_loss.item()

    def val_model(self, **kwargs):
        """
        Validates the model using the validation dataset.

        Parameters:
            **kwargs: Arbitrary keyword arguments including:
                - val_image (torch.Tensor): The batch of validation images.
                - val_mask (torch.Tensor): The batch of ground truth masks for validation images.

        Returns:
            float: The validation loss for the current step.
        """
        self.optimizer.zero_grad()

        predicted_mask = self.model(kwargs["val_image"])
        val_loss = self.dice_loss(predicted_mask, kwargs["val_mask"])

        return val_loss.item()

    def save_checkpoints(self, **kwargs):
        """
        Saves model checkpoints during the training process. Different checkpoint strategies can be applied
        based on the epoch number or validation performance.

        Parameters:
            **kwargs: Arbitrary keyword arguments including:
                - epoch (int): The current epoch number.

        Raises:
            Exception: If the checkpoints path cannot be found.
        """
        if os.path.join(CHECKPOINTS_PATH):
            if self.epochs != kwargs["epoch"]:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        CHECKPOINTS_PATH, f"model_{kwargs['epoch']}.pth"
                    ).capitalize(),
                )
            else:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(BEST_MODEL_PATH, "last_model.pth"),
                )
        else:
            raise Exception("Checkpoints path cannot be found".capitalize())

    def show_progress(self, **kwargs):
        """
        Displays or logs the training progress after each epoch. It can show detailed progress including
        loss values or a simple message indicating the completion of an epoch.

        Parameters:
            **kwargs: Arbitrary keyword arguments including:
                - epoch (int): The current epoch number.
                - train_loss (float): The average training loss for the current epoch.
                - val_loss (float): The average validation loss for the current epoch.
        """
        if self.display:
            print(
                f"Epoch: {kwargs['epoch']}, Train Loss: {kwargs['train_loss']}, Val Loss: {kwargs['val_loss']}"
            )
            logging.info(
                f"Epoch: {kwargs['epoch']}, Train Loss: {kwargs['train_loss']}, Val Loss: {kwargs['val_loss']}"
            )
        else:
            print(f"Epoch: {kwargs['epoch']} is done".capitalize())
            logging.info(f"Epoch: {kwargs['epoch']} is done".capitalize())

    def train(self):
        self.__setup__()

        for epoch in range(self.epochs):
            total_train_loss = list()
            total_val_loss = list()
            for image, mask in self.train_dataloader:
                train_image = image.to(self.device)
                train_mask = mask.to(self.device)

                train_loss = self.train_model(
                    train_image=train_image, train_mask=train_mask
                )
                total_train_loss.append(train_loss)

            for image, mask in self.val_dataloader:
                val_image = image.to(self.device)
                val_mask = mask.to(self.device)

                val_loss = self.val_model(val_image=val_image, val_mask=val_mask)
                total_val_loss.append(val_loss)

            self.history["train_loss"].append(np.mean(total_train_loss))
            self.history["val_loss"].append(np.mean(total_val_loss))

            try:
                self.save_checkpoints(epoch=epoch + 1)
            except Exception as e:
                raise Exception("Checkpoints could not be saved".capitalize())
            else:
                images, _ = next(iter(self.train_dataloader))
                images = images.to(self.device)
                generated_masks = self.model(images)
                save_image(
                    generated_masks,
                    os.path.join(IMAGES_PATH, "images_{}.png".format(epoch + 1)),
                    normalize=True,
                )
            finally:
                self.show_progress(
                    epoch=epoch,
                    train_loss=np.mean(total_train_loss),
                    val_loss=np.mean(total_val_loss),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the training for U-Net".title()
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

    args = parser.parse_args()

    if args.device == "mps":
        if (
            args.smooth_value
            and args.epochs
            and args.learning_rate
            and args.beta1
            and args.beta2
            and args.device
            and args.display
        ):
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
    else:
        raise Exception("Device is not supported".capitalize())
