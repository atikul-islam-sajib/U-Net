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

from config import PROCESSED_PATH, CHECKPOINTS_PATH, IMAGES_PATH
from utils import device_init, weight_init, config
from UNet import UNet


class Trainer:
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
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
            )
            self.criterion = nn.BCELoss()
        finally:
            if os.path.exists(PROCESSED_PATH):
                self.train_dataloader = os.path.join(
                    PROCESSED_PATH, "train_dataloader.pt"
                )
                self.val_dataloader = os.path.join(PROCESSED_PATH, "val_dataloader.pt")
            else:
                raise Exception("Processed path cannot be found".capitalize())

    def l2_regularization(self, model):
        return sum(torch.norm(params, p=2) for params in model.parameters())

    def dice_loss(self, predicted, target):
        predicted = predicted.view(-1)
        target = target.view(-1)
        intersection = (predicted * target).sum() + self.smooth_value
        return (2.0 * intersection) / (
            predicted.sum() + target.sum() + self.smooth_value
        )

    def train_model(self, **kwargs):
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
        self.optimizer.zero_grad()

        predicted_mask = self.model(kwargs["val_image"])
        val_loss = self.dice_loss(predicted_mask, kwargs["val_mask"])

        return val_loss.item()

    def save_checkpoints(self, **kwargs):
        if os.path.join(CHECKPOINTS_PATH):
            if self.epochs != kwargs["epoch"]:
                torch.save(
                    self.model,
                    os.path.join(
                        CHECKPOINTS_PATH, f"model_{kwargs['epoch']}.pth"
                    ).capitalize(),
                )
            else:
                torch.save(self.model, os.path.join(CHECKPOINTS_PATH, "last_model.pth"))
        else:
            raise Exception("Checkpoints path cannot be found".capitalize())

    def show_progress(self, **kwargs):
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
            for _, (image, mask) in enumerate(self.train_dataloader):
                train_image = image.to(self.device)
                train_mask = mask.to(self.device)

                train_loss = self.train_model(
                    train_image=train_image, train_mask=train_mask
                )
                total_train_loss.append(train_loss)

            for _, (image, mask) in enumerate(self.val_dataloader):
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
                    nrows=2,
                )
            finally:
                self.show_progress(
                    epoch=epoch,
                    train_loss=np.mean(total_train_loss),
                    val_loss=np.mean(total_val_loss),
                )
