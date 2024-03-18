# Import all libraries
import os
import yaml
import joblib


def clean(filename):
    if filename:
        for file in os.listdir(filename):
            os.remove(os.path.join(filename, file))
    else:
        raise ValueError("File not found".capitalize())


def config():
    with open("./deafult_params.yml", "r") as file:
        return yaml.safe_load(file)


def load_pickle(value=None, filename=None):
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)
