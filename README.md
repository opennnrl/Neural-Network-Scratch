# Neural Network
## Neural Network Python Scratch for Research

# MNIST Neural Network (Fully Connected Network)

This repository contains an implementation of a fully connected neural network using NumPy, and OpenCV to classify MNIST handwritten digits. The model uses multiple hidden layers with ReLU activation and a final softmax output layer.

## Features
- Uses MNIST dataset with downsampled 8x8 images.
- Implements a custom neural network class.
- Uses floating point 16 (FP16) for reduced memory usage.
- Trains using mini-batch gradient descent.
- Includes visualization of training and test errors.

## Requirements
Ensure you have the following installed:
```bash
pip install numpy opencv-python tensorflow matplotlib scikit-learn
```

## Usage
Run the script to train the model:
```bash
python mnist_nn.py
```

## Code Overview
### 1. Data Preprocessing
- Loads MNIST dataset from Keras.
- Resizes images to 8x8.
- Normalizes pixel values.
- Converts labels to one-hot encoding.

### 2. Neural Network Implementation
- `MiddleLayer`: Implements ReLU activation.
- `OutputLayer`: Implements softmax activation.
- `NeuralNetwork`: Handles forward and backward propagation, loss calculation, and parameter updates.

### 3. Training
- Trains for 50 epochs with a batch size of 128.
- Uses cross-entropy loss and accuracy for evaluation.
- Plots training and test error.

## Example Output
```bash
Epochs: 1/50   Error Train: 1.23   Error Test: 1.25
...
Epochs: 50/50  Error Train: 0.12   Error Test: 0.15
TRAIN ACCURACY:  98.5%
TEST ACCURACY:  97.8%
Training time: 35.2 seconds
```
