# Driver Drowsiness Detection

## Overview
This project develops a deep learning model to detect driver drowsiness using facial image analysis. The system classifies driver states as either awake or drowsy to help prevent fatigue-related accidents.

## Features
- Binary classification of driver alertness (Awake/Drowsy)
- Uses MobileNetV2 as a pre-trained base model for feature extraction
- Incorporates an LSTM layer for sequence processing
- Accuracy and performance metrics
  
## Requirements
- Python 3.8+
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- splitfolders

## Installation
```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn splitfolders
```

## Dataset
The model is trained on the [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/data) from Kaggle, containing facial images of drivers in awake and drowsy state.

## Model Architecture
- Base Model: MobileNetV2
- Added Layers: 
  - Global Average Pooling
  - Dense Layers
  - Batch Normalization
  - Dropout
  - LSTM Layer

## Usage
1. Download the dataset from Kaggle
2. Prepare your image dataset
3. Run the preprocessing script
4. Train the model
5. Evaluate model performance

## Results
The model provides:
- Binary classification of driver state
- Confusion matrix
- Accuracy and performance metrics

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments
- [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/data)
