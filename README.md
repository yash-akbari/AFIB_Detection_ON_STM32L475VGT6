AFIB_Detection_ON_STM32L475VGT6

his repository provides an end-to-end solution for detecting Atrial Fibrillation (AFib) using a BiLSTM-based model. The project focuses on training the model with RR interval data, optimizing it for low power consumption on the STM32L475VGT6 microcontroller.

Table of Contents
Prerequisites
Project Setup
Training the Model
Optimizing the Model for STM32
Deployment to STM32
Power Consumption Measurements
Troubleshooting
Prerequisites
Before getting started, ensure that you have the following tools installed:

STM32CubeIDE: For development and debugging on STM32 microcontrollers.
ST Cube AI: To optimize TensorFlow/Keras models for STM32 MCUs.
Python 3.11: For training the model using TensorFlow.
TensorFlow 2.15: The version used for model training.
Git: To clone the repository.
STM32L475VGT6: The target microcontroller for deploying the trained model.
Project Setup
Step 1: Clone the Repository
Clone the repository to your local machine

Step 2: Dataset
The dataset is provided in .npz format and is already included in the repository. The dataset contains RR interval data and binary labels for Atrial Fibrillation detection.

The RR interval data consists of 40 beats with a 39-beat overlap.
The labels are binary: 0 for no AFib and 1 for AFib.
You can find the dataset in the data/ folder. The .npz file is already preprocessed and split for use in training the model.

If you need to preprocess or modify the dataset, you can follow the steps in the data_processing.py script included in the repository.

Training the Model
Load the Preprocessed Dataset: After setting up the repository and installing dependencies, the dataset will be ready to be loaded for training.

Train the BiLSTM Model: The model architecture is already defined in the repository. It uses a Bidirectional LSTM layer followed by a Dense output layer for classification.

Evaluate the Model: The model’s performance can be evaluated on a test set to ensure it meets the accuracy requirements. The model should achieve an accuracy of around 97% when trained on the dataset.

Optimizing the Model for STM32
Once the model is trained, it needs to be optimized for deployment on STM32:

Upload the Model to ST Cube AI: Use ST Cube AI to convert the trained model into a format compatible with STM32 microcontrollers. During this step, make sure not to apply quantization (unless required).

Generate the STM32 Cube Project: ST Cube AI will generate an STM32 Cube project. This project includes the necessary code to deploy the model on the STM32L475VGT6 MCU.

Deployment to STM32
Setup STM32CubeIDE: Open the generated STM32Cube project in STM32CubeIDE. Make sure to disable unnecessary peripherals to optimize the system for low power usage.

Modify the Project: Replace any dummy or random data inputs with real AFib data (RR intervals) for testing the model.

Flash the STM32: Flash the microcontroller with the generated firmware, then test the inference capabilities using UART for data transfer between the PC and the MCU.

Power Consumption Measurements
During testing, the power consumption was measured with the ST-LINK V3PWR board. The current consumption at 1.8V was recorded at 13,000 μA (13 mA).

Future testing may explore optimizing further using low-power modes or other methods to reduce the overall power consumption.

Troubleshooting
Incorrect Power Consumption: If the power consumption is higher than expected, ensure that all unnecessary peripherals are disabled in STM32CubeMX and STM32CubeIDE.
Inference Accuracy Drops: Check the input data to ensure it's correctly formatted, and review the model training parameters for potential adjustments.
UART Communication Issues: If UART communication fails, ensure that the data format (e.g., fixed-length data) is correctly aligned between the PC and STM32.
