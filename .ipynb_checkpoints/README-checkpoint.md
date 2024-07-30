# Employee Attrition and Department Prediction Model

This repository contains a neural network model designed to predict employee attrition and department assignments based on various input features. The model uses TensorFlow and Keras for implementation and provides separate branches for the two prediction tasks.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Data Preparation](#data-preparation)


## Overview

This project aims to predict employee attrition (whether an employee will leave the company) and department assignments using a shared neural network model. The model is designed with separate branches for each prediction task, utilizing shared layers to leverage common features.

## Model Architecture

The model consists of the following components:

1. **Input Layer**: Accepts features from the training data.
2. **Shared Layers**: Common layers shared between both prediction tasks.
3. **Branch for Department Prediction**:
    - Hidden Layer
    - Output Layer with softmax activation
4. **Branch for Attrition Prediction**:
    - Hidden Layer
    - Output Layer with sigmoid activation

### Activation Functions

- **Sigmoid**: Used for the attrition output layer to produce probabilities between 0 and 1 for binary classification.
- **Softmax**: Used for the department output layer to produce a probability distribution over multiple classes for multiclass classification.

## Data Preparation

Ensure your data is preprocessed correctly before training the model. The target variables should be properly encoded, and the input features should be normalized or standardized if necessary.

Example of loading and preparing data:

```python
import numpy as np
import pandas as pd

# Load your data
X_train = np.load('X_train.npy')
y_train_department = np.load('y_train_department.npy')
y_train_attrition = np.load('y_train_attrition.npy')

X_test = np.load('X_test.npy')
y_test_department = np.load('y_test_department.npy')
y_test_attrition = np.load('y_test_attrition.npy')
