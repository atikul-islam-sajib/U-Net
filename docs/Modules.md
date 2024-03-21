
## U-Net Model Training and Testing - Import Modules

### Overview

This guide provides step-by-step instructions for loading a dataset, training the U-Net model, and testing the model's performance on semantic segmentation tasks. It also covers how to visualize the training process and prediction results.

### Requirements

- Python 3.9 or newer
- Access to a CUDA-compatible GPU (if using CUDA for training and testing)
- Required Python libraries: `torch`, `matplotlib`, `numpy`, `PIL`

### Steps

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