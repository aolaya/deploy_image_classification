# Image Classification Model and Deployment
<p align="center">
    <img src="Images/docker-logo-6D6F987702-seeklogo.com.png" width="10%" valign="middle" alt="Docker">
    <img src="Images/plus_sign.svg" width="7%" valign="middle" alt="Plus">
    <img src="Images/FastAPI_logo.svg.png" width="15%" valign="middle" alt="FastAPI">
    <img src="Images/plus_sign.svg" width="7%" valign="middle" alt="Plus">
    <img src="Images/Microsoft_Azure-Logo.wine.png" width="15%" valign="middle" alt="Azure">
    <img src="Images/plus_sign.svg" width="7%" valign="middle" alt="Plus">
    <img src="Images/poetry_logo.jpeg" width="5%" valign="middle" alt="Poetry">
    <img src="Images/plus_sign.svg" width="7%" valign="middle" alt="Plus">
    <img src="Images\TensorFlow_logo.svg.png" width="15%" valign="middle" alt="Tensorflow">
</p>

## Overview
An advanced computer vision system for classifying environmental and urban scenes using a multi-modal Convolutional Neural Network (CNN) approach combining raw image data with engineered features (HOG, HSV, ResNet embeddings). The model achieves 81.93% accuracy across six distinct categories: Buildings, Forest, Glacier, Mountain, Sea, and Street.

**Try Out the Classification Model:** https://image-classification-ao-2314.azurewebsites.net/ 

## Applications
- Environmental monitoring and landscape change tracking
- Urban development analysis
- Large-scale image organization and search
- Computer vision research and development

## About the Dataset

<p align="center">
    <!--SampleImages-->
      <img src="Images\sample_images.png"  width=90% alt="SampleImages">
  
This dataset comprises of 25,000 images, each with a resolution of 150x150 pixels, divided into six categories: Buildings, Forest, Glacier, Mountain, Sea, and Street. The data is organized into separate zip files for training, testing, and prediction, with around 14,000 images in the training set, 3,000 in the test set, and 7,000 for prediction. In the training set, each feature has roughly 2,300 examples. In the test set, each feature has 500 examples.

## Preprocessing  
To prepare the dataset for analysis, several preprocessing steps were undertaken to standardize and optimize the input data. First, all images were resized to a uniform resolution of 150x150 pixels, ensuring consistent dimensions across the dataset. Next, pixel values were normalized by scaling them to a range between 0 and 1, a technique that improves numerical stability during training. Finally, a train-test split was applied, with training and test sets loaded separately to prevent data leakage. From the training set, a validation subset was further extracted to support hyperparameter tuning later on during the model training phase.

## Model Performance Comparison
<div align="center">

| Model | Test Accuracy | Training Time | Classification Time/Image |
|-------|--------------|---------------|-------------------------|
| CNN Naïve | 79.70% | 13m 33.6s | 0.002034s |
| CNN Multimodal | 81.93% | 13m 37.5s | 0.001954s |
| ResNet50 Naïve | 69.60% | 54m 47.4s | 0.015857s |
| ResNet50 Multimodal | 79.80% | 58m 24.6s | 0.017395s |
</div>

## Key Findings
- Multi-modal approach significantly improved classification accuracy
- Outperformed more complex architectures like ResNet50
- Demonstrated robust generalization across diverse scene types
- Achieved balanced performance across all categories

## Future Improvements
- Enhanced data augmentation strategies
- Fine-tuning of ResNet architecture
- Integration of attention mechanisms
- Expanded category coverage

##### Source: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
