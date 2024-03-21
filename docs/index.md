# U-Net for Semantic Image Segmentation

U-Net is a convolutional neural network designed for semantic image segmentation. This implementation of U-Net is tailored for high performance on various image segmentation tasks, allowing for precise object localization within images.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">



## Features

| Feature                      | Description                                                                                                                                                      |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Efficient Implementation** | Utilizes an optimized U-Net model architecture for superior performance on diverse image segmentation tasks.                                                     |
| **Custom Dataset Support**   | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                        |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                         |
| **Visualization Tools**      | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                        |
| **Custom Training via CLI**  | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                    |
| **Import Modules**           | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of U-Net functionality.  |
| **Multi-Platform Support**   | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |


## Getting Started

To present the requirements for your U-Net implementation in a table format for clarity and structure in your README, you can use the following markdown representation:

---

## Requirements

| Requirement                 | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| **Python Version**          | Python 3.9 or newer is required for compatibility with the latest features and library support. |
| **CUDA-compatible GPU**     | Access to a CUDA-compatible GPU is recommended for training and testing with CUDA acceleration. |
| **Python Libraries**        | Essential libraries include: **torch**, **matplotlib**, **numpy**, **PIL**, **scikit-learn**, **opencv-python** |


## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction | Command |
|------|-------------|---------|
| 1 | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/U-Net.git** |
| 2 | Navigate into the project directory. | **cd U-Net** |
| 3 | Install the required Python packages. | **pip install -r requirements.txt** |


## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the U-Net model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
    - Stores model checkpoints during training for later resumption.
  
- **best_model/**
    - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**
    - Houses all model checkpoints generated throughout the training process.

- **data/**
    - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
    - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**
    - **Log** files for debugging and tracking model training progress.

- **metrics/**
     - Files related to model performance metrics for evaluation purposes.

- **outputs/**
    - **test_images/**: Images generated during the testing phase, including segmentation outputs.
    - **train_gif/**: GIFs compiled from training images showcasing the model's learning progress.
    - **train_images/**: Images generated during training for performance visualization.

- **research/**
     - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**
     - Source code directory containing all custom modules, scripts, and utility functions for the U-Net model.

- **unittest/**
    - Unit tests ensuring code reliability, correctness, and functionality across various project components.


### Dataset Organization for Semantic Image Segmentation

The dataset is organized into three categories for semantic image segmentation tasks: benign, normal, and malignant. Each category directly contains paired images and their corresponding segmentation masks, stored together to simplify the association between images and masks.

#### Directory Structure:

```
segmentation/
├── benign/
│   ├── benign(1).png
│   ├── benign(1)_mask.png
│   ├── benign(2).png
│   ├── benign(2)_mask.png
│   ├── ...
├── normal/
│   ├── normal(1).png
│   ├── normal(1)_mask.png
│   ├── normal(2).png
│   ├── normal(2)_mask.png
│   ├── ...
├── malignant/
│   ├── malignant(1).png
│   ├── malignant(1)_mask.png
│   ├── malignant(2).png
│   ├── malignant(2)_mask.png
│   ├── ...
```

#### Naming Convention:

- **Images and Masks**: Within each category folder, images and their corresponding masks are stored together. The naming convention for images is `[category](n).png`, and for masks, it is `[category](n)_mask.png`, where `[category]` represents the type of the image (benign, normal, or malignant), and `(n)` is a unique identifier. This convention facilitates easy identification and association of each image with its respective mask.


For detailed documentation on the dataset visit the [Dataset - Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).




#### Training the Model

To train the model, run the following command, specifying the path to your dataset and other training parameters:

```
python src/cli.py --image_path /content/semantic.zip --batch_size 4 --device cuda --smooth_value 0.01 --epochs 100 --learning_rate 0.0002 --display False --train
```

#### Testing the Model

After training, you can test the model on your test dataset by running:

```
python src/cli.py --test --device cuda --samples 20
```

### Visualization

Visualize the training process and predictions by checking the generated plots in the `outputs` directory.

## Customization

You can customize various aspects of the training and model by modifying the arguments passed to `src/cli.py`. For a full list of customizable parameters:

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


#### 1. Import Necessary Modules

First, ensure that you have the necessary modules available in your Python environment. These modules include functionalities for data loading, model definition, training, and evaluation.

```python
from src.dataloader import Loader
from src.UNet import UNet
from src.trainer import Trainer
from src.test import Charts
```

#### 2. Load the Dataset

Use the `Loader` class to load your dataset. Specify the path to your dataset and the desired batch size. This example demonstrates loading a dataset from a zipped file and creating a DataLoader object.

```python
loader = Loader(image_path="/content/semantic.zip", batch_size=4)
loader.unzip_folder()
dataloader = loader.create_dataloader()
```

#### 3. Train the Model

Initialize the `Trainer` class with training parameters such as the number of epochs, smooth value, learning rate, and the device on which the training is to be performed. Then, start the training process.

```python
trainer = Trainer(epochs=100,
                  smooth_value=0.01,
                  learning_rate=0.0002,
                  device="cuda",  # Use "cpu" if CUDA is not available
                  display=True)

trainer.train()
```

The training process outputs the training and validation losses for each epoch, providing insight into the model's learning progress.

#### 4. Test the Model

After training, evaluate the model's performance on the test dataset using the `Charts` class. This class also generates visualizations for the predictions and loss curves.

```python
chart = Charts(samples=20, device="cuda")  # Use "cpu" if CUDA is not available
chart.test()
```

#### 5. Visualize Results

Visualize the test results and the loss curves by displaying the generated images. Ensure you specify the correct paths to the images.

```python
from IPython.display import Image

# Display the result image
Image("/content/U-Net/outputs/test_images/result.png")

# Display the loss curve image
Image("/content/U-Net/outputs/test_images/loss.png")
```

## Contributing

Contributions to improve this implementation of U-Net are welcome. Please follow the standard fork-branch-pull request workflow.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.