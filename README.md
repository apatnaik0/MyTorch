# MyTorch

**MyTorch** is a custom deep-learning library built from scratch, inspired by PyTorch. 

This project is a personal initiative to deeply understand the internals of deep learning frameworks. By implementing core components—forward propagation, backpropagation, loss calculation, and optimization algorithms—I aim to bridge the gap between mathematical theory and efficient software implementation.

## Project Overview

In this initial phase, the library focuses on the fundamental building blocks of **Multilayer Perceptrons (MLPs)**. The goal is to create a modular and extensible framework where components like layers, activation functions, and optimizers can be easily composed to build and train neural networks.

## Directory Structure

```
MyTorch/
├── models/
│   └── mlp.py            # Implementation of a Multilayer Perceptron
├── mytorch/
│   ├── nn/
│   │   ├── activation.py # Activation functions (Sigmoid, Tanh, ReLU, etc.)
│   │   ├── batchnorm.py  # Batch Normalization implementation
│   │   ├── linear.py     # Linear (Fully Connected) Layer
│   │   └── loss.py       # Loss functions (MSE, CrossEntropy, etc.)
│   └── optim/
│       └── sgd.py        # Stochastic Gradient Descent Optimizer
└── requirements.txt      # Project dependencies
```

## Key Components

### 1. `mytorch.nn`
This module contains the core neural network layers and operations:
- **`Linear`**: Implements a fully connected layer with learnable weights and biases. Handles forward and backward passes.
- **`Activation`**: Includes standard activation functions like ReLU, Sigmoid, etc., responsible for introducing non-linearity.
- **`BatchNorm`**: Implements Batch Normalization to stabilize and accelerate training.
- **`Loss`**: specific loss functions to measure the performance of the model against targets.

### 2. `mytorch.optim`
- **`SGD`**: Implements the Stochastic Gradient Descent optimization algorithm to update model parameters based on computed gradients.

### 3. `models`
- **`MLP`**: A reference implementation utilizing the custom `mytorch` components to construct a complete Multilayer Perceptron.

## Future Roadmap

This project is part of a larger planned series of implementations to cover the spectrum of modern deep learning:

- **Convolutional Neural Networks (CNNs)**: Implementing convolution operations, pooling, and image processing capabilities.
- **Recurrent Neural Networks (RNNs)**: Building standard RNNs, identifying vanishing gradient problems.
- **Gated Architectures**: Implementing **GRUs** (Gated Recurrent Units) and **LSTMs** (Long Short-Term Memory) to handle long-range dependencies in sequence data.

## Installation & Usage

### Prerequisites
Ensure you have Python installed. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project
You can explore the MLP implementation by running the model script:

```bash
python models/mlp.py
```

---
*This project is a personal learning endeavor and is not intended for production usage.*
