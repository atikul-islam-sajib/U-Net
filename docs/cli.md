## U-Net CLI Tool Documentation

### Overview

This document outlines the usage of the CLI tool designed for training and testing the U-Net model for image segmentation tasks. The tool provides a command-line interface for easy interaction, allowing users to specify training parameters, conduct model training, and evaluate model performance through testing.

### Usage

To use this CLI tool, execute the `cli.py` script from the command line, followed by the appropriate arguments to specify your operation (train or test) and the parameters for the operation.

```
python src/cli.py [arguments]
```

### Command-Line Arguments

The table below lists all the command-line arguments available in the CLI tool, their descriptions, and whether they are required.

| Argument          | Description                                         | Required | Default |
|-------------------|-----------------------------------------------------|:--------:|:-------:|
| `--image_path`    | Path to the image folder                            | Yes      | N/A     |
| `--batch_size`    | Batch size for the DataLoader                       | Yes      | N/A     |
| `--smooth_value`  | Smooth value for model training                     | No       | `0.01`  |
| `--epochs`        | Number of epochs for training                       | No       | `100`   |
| `--learning_rate` | Learning rate for the optimizer                     | No       | `0.0002`|
| `--beta1`         | Beta1 value for Adam optimizer                      | No       | `0.5`   |
| `--beta2`         | Beta2 value for Adam optimizer                      | No       | `0.999` |
| `--device`        | Device to run the model on (e.g., `cpu`, `mps`)     | No       | `mps`   |
| `--l2`            | Apply L2 regularization                             | No       | `False` |
| `--criterion`     | Apply a criterion for training                      | No       | `False` |
| `--display`       | Display training progress                           | No       | `True`  |
| `--samples`       | Number of samples to plot, choices: 10, 20          | No       | `20`    |
| `--train`         | Flag to indicate training mode                      | No       | N/A     |
| `--test`          | Flag to indicate testing mode                       | No       | N/A     |

### Training the Model using CUDA

To train the model, you must specify the `--train` flag along with the required parameters `--image_path` and `--batch_size`. Optional parameters can also be provided to customize the training process. Here's an example command:

```
python src/cli.py --image_path /content/semantic.zip --batch_size 4 --device cuda --smooth_value 0.01 --epochs 100 --learning_rate 0.0002 --display False --train
```

### Training the Model using MPS

To train the model, you must specify the `--train` flag along with the required parameters `--image_path` and `--batch_size`. Optional parameters can also be provided to customize the training process. Here's an example command:

```
python src/cli.py --image_path /content/semantic.zip --batch_size 4 --device mps --smooth_value 0.01 --epochs 100 --learning_rate 0.0002 --display False --train
```

### Training the Model using CPU

To train the model, you must specify the `--train` flag along with the required parameters `--image_path` and `--batch_size`. Optional parameters can also be provided to customize the training process. Here's an example command:

```
python src/cli.py --image_path /content/semantic.zip --batch_size 4 --device cpu --smooth_value 0.01 --epochs 100 --learning_rate 0.0002 --display False --train
```

### Testing the Model CUDA

To test the model, use the `--test` flag. Specify the `--device` and `--samples` to customize the testing process. Note that `--image_path` and `--batch_size` are not required for testing:

```
python src/cli.py --test --device cuda --samples 20
```

### Testing the Model MPS

To test the model, use the `--test` flag. Specify the `--device` and `--samples` to customize the testing process. Note that `--image_path` and `--batch_size` are not required for testing:

```
python src/cli.py --test --device mps --samples 20
```

### Testing the Model CPU

To test the model, use the `--test` flag. Specify the `--device` and `--samples` to customize the testing process. Note that `--image_path` and `--batch_size` are not required for testing:

```
python src/cli.py --test --device cpu --samples 20
```

### Additional Notes

- Ensure that the paths provided to `--image_path` are correct and accessible.
- Adjust the batch size according to your system's memory capabilities.
- The `--device` option allows you to select between different computing devices, such as CPU or GPU (MPS for Mac), to optimize performance.
