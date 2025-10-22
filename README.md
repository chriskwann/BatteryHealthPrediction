# Battery Health Prediction using Neural Networks
## Overview
This project demonstrates how to use a feedforward neural network (ANN) to predict battery health scores based on several measurable parameters such as temperature, charge cycles, and efficiency.
The goal is to model a relationship between input parameters and the resulting battery health score using TensorFlow and Keras.

## Libraries Used in Python  
1.NumPy and Pandas – for numerical data handling  
2.Matplotlib – for plotting graphs  
3.Scikit-learn – for preprocessing and evaluation  
4.TensorFlow / Keras – for building and training the neural network  

## How It Works
Dataset Preparation  
A small dataset is manually defined for training and testing.  
Each row represents a battery sample with input parameters:  
Temperature / Age  
Charge cycles  
Internal resistance  
Efficiency  
Target: Health Score  
Feature Scaling  
Input data is normalized using MinMaxScaler to improve model performance.  

## Model Architecture
4 input features  
2 hidden layers:  
16 neurons (ReLU activation)  
8 neurons (ReLU activation)  
1 output neuron (linear activation)  

## Training
Optimizer: adam  
Loss function: Mean Squared Error  
100 epochs of training  

## Evaluation
Model is evaluated using R² Score.  
Visualization  
Training loss curve  
True vs Predicted battery health comparison  
