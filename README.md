# CNN Image Classification with Model Deployment (TF, TF-Lite, TFJS)

## Overview
This project implements a CNN-based solution for multi-class image classification, showcasing an end-to-end deep learning workflow. 
From dataset preprocessing and augmentation to model training and evaluation, every step is designed to build a robust model. 
The trained model is then exported in TensorFlow SavedModel, TF-Lite, and TensorFlow.js formats, allowing easy deployment across web and mobile platforms.
Key aspects of the project:
- **Preprocessing**: Load and preprocess images, including resizing, normalization, and augmentation to increase model robustness.
- **Model Training**: Build a Convolutional Neural Network (CNN) using TensorFlow, train it on the dataset with callbacks, and monitor performance through accuracy and loss metrics.
- **Evaluation**: Validate the model on unseen test data to ensure generalization and assess performance.
- **Deployment**: Export the trained model in TensorFlow SavedModel, TF-Lite, and TensorFlow.js formats for flexible deployment across platforms.
- **Inference**: Perform predictions on new, unseen images to demonstrate practical application and model reliability.

---

## Dataset
The dataset consists of flower images for multi-class classification, sourced from Kaggle. 
It contains images of various categories with different resolutions, preprocessed and split into training, validation, and testing sets.

Key details of the dataset:
| Feature       | Description                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------- |
| Image         | Individual image file used for classification                                                     |
| Label         | Class/category of the image (target variable)                                                     |
| Resolution    | Original resolution of the image before preprocessing                                             |
| Augmented     | Indicator if the image was generated through augmentation techniques (e.g., rotation, flip, zoom) |
| Dataset Split | Indicates whether the image is part of Training, Validation, or Test set                          |

**Data source:** [Flower Image Dataset on Kaggle](https://www.kaggle.com/datasets/chethannagesh03/flower?select=train)

---

## How to Run
1. Clone this repository or download the files.
2. Install dependencies:
Make sure to use a fresh Python environment. Then run:
`pip install -r requirements.txt`
3. Open and run the notebook:
Open `KlasifikasiGambar_Imelda.ipynb` using Jupyter Notebook or Google Colab.

## Model Formats
- **SavedModel** (saved_model/)

  TensorFlow’s default format, suitable for deployment or retraining.
  
- **TensorFlow Lite** (tflite/model.tflite)

  Lightweight format optimized for mobile and IoT devices.
  
- **TensorFlow.js** (tfjs_model/)

  Format for running the model directly in web browsers.

  ---


  
