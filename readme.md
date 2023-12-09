# Polyp Segmentation using UNet++ for Colonoscopy Images

## Business Context
Machine learning and deep learning technologies have made significant strides in healthcare and medical sciences. This project focuses on using such technologies for polyp recognition and segmentation in colonoscopy images, aiding medical professionals in their diagnosis.

---

## Data Overview
The CVC-Clinic database comprises frames extracted from colonoscopy videos. The dataset includes polyp frames and corresponding ground truth, represented as masks in both .png and .tiff formats.

---

## Aim

To implement polyp recognition and segmentation for colonoscopy images using UNet++ model.

---

## Tech Stack
- Language: `Python`
- Deep learning library: `PyTorch`
- Computer vision library: `OpenCV`
- Other Python libraries: `scikit-learn`, `pandas`, `numpy`, `albumentations`, etc.

---

## Approach
1. **Data Understanding**: Explore and understand the dataset.
2. **Understanding Evaluation Metrics**: Familiarize with the metrics used for model evaluation.
3. **UNet Architecture**: Understand the UNet architecture and its relevance in medical science applications.
4. **UNet++**: Learn about UNet++ and how it differs from the standard UNet.
5. **Environment Setup**: Prepare the working environment for the project.
6. **Data Augmentation**: Generate augmented data to enhance model performance.
7. **Model Building**: Develop the UNet++ model using PyTorch.
8. **Model Training**: Train the model (Note: GPU is recommended for faster training).
9. **Model Prediction**: Make predictions using the trained model.

---

## Modular Code Overview

1. **input**: Contains data folders (PNG and TIF) with colonoscopy images in different formats.
2. **src**: The core of the project with modularized code, including:
   - `ML_pipeline`
   - `engine.py`
   - `config.yaml`: Config file for project constants.
3. **output**: Contains the trained model (reusable file) and predicted images.
4. **lib**: A reference folder with the original IPython notebook.
5. `requirements.txt`: Lists all the packages and libraries used in the project.

---

## Getting Started

1. Create a new env.

2. To install the dependencies run:
	```
	pip install -r requirements.txt
	```

3. run the `engine.py` file

---
