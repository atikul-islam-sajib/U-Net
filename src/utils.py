# Import all libraries
import os
import yaml
import joblib
import torch
import torch.nn as nn


def clean(filename):
    """
    Removes all files within the specified directory.

    Parameters:
    - filename (str): The path to the directory from which files will be removed.

    Raises:
    - ValueError: If the directory specified does not exist.
    """
    if filename:
        for file in os.listdir(filename):
            os.remove(os.path.join(filename, file))
    else:
        raise ValueError("File not found".capitalize())


def config():
    """
    Loads configuration parameters from a YAML file.

    Returns:
    - dict: A dictionary containing configuration parameters.
    """
    with open("./deafult_params.yml", "r") as file:
        return yaml.safe_load(file)


def dump_pickle(value=None, filename=None):
    """
    Serializes and saves a Python object to a file using joblib.

    Parameters:
    - value: The Python object to serialize.
    - filename (str): The path to the file where the object will be saved.
    """
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)


def load_pickle(filename):
    """
    Loads a Python object from a pickle file.

    Parameters:
    - filename (str): The path to the pickle file.

    Returns:
    - The Python object loaded from the file.
    """
    if filename:
        return joblib.load(filename=filename)


def weight_init(m):
    """
    Initializes the weights of Convolutional and BatchNorm layers following a normal distribution.

    Parameters:
    - m: A PyTorch module.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def device_init(device="mps"):
    """
    Initializes the device for PyTorch operations.

    Parameters:
    - device (str): The type of device to use ('mps', 'cuda', 'cpu').

    Returns:
    - torch.device: The initialized device.
    """
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")
