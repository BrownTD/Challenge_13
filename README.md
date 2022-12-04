# Challenge_13

For this challenge my role is to act as a risk management associate at a venture capital firm. I am tasked with creating a model that predicts whether applicants will be successfu if funded by the firm. The start of this analysis comes from a csv file that contains more than 34,000 organizations that have recieved funding from the firm, and it contains information that includes whether or not it became successful. I then use my machine learning knowledge to create a binary classifier model that will predict whether an applicant will become a successful business. This challenge consists of three technical deliverables, we are to preprocess data for a neural network model, use the model-fit-predict pattern to compile and evaluate a binary classification model, and attempt to optimize the model two times.

The changes I made to optimize the model include:

#1. Increasing the number of neurons in the output layer to 3. I increased the number of hidden layers. I increased the number of epochs. This attempt increased the model's accuracy but only by .05%

#2. I increased the number of neurons in the output layer to 5. I increased the number of nodes per layer by 10 in each hidden layer. I decreased the number of epochs. This attempt did not increase the model's accuracy, it ended up decreasing the accuracy by almost .2%

# Technologies

This project utilizes python 3.7 with the following packages:
- Pandas 

- scikit-learn  

- TensorFlow  

# Installation Guide

Install the following dependencies before running the application

import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

# Contributors

Nevyn Brown
