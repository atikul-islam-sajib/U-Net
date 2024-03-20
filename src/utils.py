# Import all libraries
import os
import yaml
import joblib
import torch
import torch.nn as nn


def clean(filename):
    if filename:
        for file in os.listdir(filename):
            os.remove(os.path.join(filename, file))
    else:
        raise ValueError("File not found".capitalize())


def config():
    with open("./deafult_params.yml", "r") as file:
        return yaml.safe_load(file)


def dump_pickle(value=None, filename=None):
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)


def load_pickle(filename):
    if filename:
        return joblib.load(filename=filename)


def weight_init(m):
    classname = m.__class.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")
