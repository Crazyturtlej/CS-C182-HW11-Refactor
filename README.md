# Scaling Laws: Learning Rate vs. Batch Size

A hands-on coding assignment exploring neural network scaling laws through empirical analysis of how learning rate should scale with batch size.

To see the cleaned **student version** (no solutions) of the problem notebook, refer to ```scaling_laws.ipynb```.

To check the cleaned **solutions version** of the problem notebook, refer to ```scaling_laws_solutions.ipynb```.

## Overview

This assignment is part of **CS182: Deep Learning** and focuses on scaling laws—empirical relationships that predict optimal hyperparameters as training conditions change.

You will investigate:

1. **Least-Squares SGD** - Derive the learning rate scaling law for simple linear regression
2. **MLP with SGD** - Extend the analysis to a two-layer neural network
3. **MLP with Adam** - Compare how adaptive optimizers change the scaling relationship

By the end, you'll understand how to predict optimal learning rates without exhaustive hyperparameter sweeps.

## Background

### Why Scaling Laws?

As neural networks grow larger, hyperparameter sweeps become impractical. Scaling laws provide a principled way to extrapolate optimal settings:

> "If I know the optimal learning rate for batch size B, what should I use for batch size 2B?"

This relationship is crucial for:
- Efficient distributed training across multiple GPUs
- Reducing computational costs of hyperparameter tuning
- Understanding the fundamental dynamics of optimization

### Key Concepts

- **Batch Size**: Number of samples used to compute each gradient update
- **Learning Rate**: Step size for gradient descent updates
- **Linear Scaling Rule**: A common heuristic where LR scales linearly with batch size
- **Square Root Scaling**: An alternative where LR scales with √(batch size)

### References

- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) (Goyal et al., 2017)
- [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) (Smith et al., 2017)
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) (McCandlish et al., 2018)

## Getting Started

### Prerequisites

- Python 3.9+
- NumPy
- Matplotlib
- PyTorch

## Assignment Structure

```
CS-C182-HW11-Refactor/
├── scaling_laws.ipynb           # Student Version
├── scaling_laws_solutions.ipynb # Reference Solutions
└── README.md                    
```

## Completing/Running the Assignment

Open the notebook, run cells sequentially, and complete the cells marked with `# TODO` comments!
