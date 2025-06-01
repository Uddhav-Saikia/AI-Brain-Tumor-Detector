# Brain Tumor Classification Using CNN

This repository presents a convolutional neural network (CNN) model designed to identify brain tumors from MRI scans. The model is developed using TensorFlow and Keras frameworks.

## About the Dataset

The data used consists of MRI brain images categorized into two classes: images showing tumors and those without tumors. The dataset contains a limited number of samples, which posed challenges for training the model effectively.

## Important Note

For smooth interaction with Jupyter notebooks, consider using online notebook viewers such as [nbviewer](https://nbviewer.jupyter.org/) if GitHub's native viewer does not render them properly.

## Data Enhancement

Given the relatively small and imbalanced dataset, data augmentation was employed to artificially increase the number of training samples and balance the classes. This approach improves the robustness of the model by exposing it to varied versions of the images.

The augmentation process expanded the dataset significantly by applying transformations like rotations, flips, and zooms, thereby creating new diverse training examples while retaining the original images.

## Image Preparation

Before feeding images into the model, they undergo several preprocessing steps:

1. Extracting the region of interest by cropping the brain area to reduce background noise.  
2. Standardizing all images to a fixed size to maintain consistent input dimensions.  
3. Scaling pixel values to a normalized range (0 to 1) to facilitate efficient learning.

## Splitting Data for Training

The dataset was divided into three parts to train and evaluate the model:

- Training set: 70%  
- Validation set: 15%  
- Test set: 15%

This partitioning ensures that the model is trained, validated, and tested on separate data.

## Model Architecture

The neural network receives images with a fixed dimension and processes them through the following layers:

- Initial padding to maintain spatial dimensions.  
- Convolutional layers with filters to extract features.  
- Batch normalization to stabilize and speed up training.  
- Activation using ReLU for non-linearity.  
- Pooling layers to reduce spatial dimensions and focus on important features.  
- A flattening layer to convert feature maps into a vector.  
- A final fully connected layer with sigmoid activation to perform binary classification.

### Design Choices

More complex pre-trained architectures were initially explored but resulted in overfitting due to the small dataset size and computational limitations. Therefore, a streamlined custom CNN was developed and trained from scratch, yielding satisfactory performance with lower resource requirements.

## Training Overview

The model was trained over multiple epochs, showing continuous improvement in accuracy and reduction in loss on validation data, with optimal performance reached before training completion.

## Performance Metrics

On the unseen test set, the model demonstrated:

- Accuracy around 89%  
- F1 score near 0.88

These results indicate that the model is capable of reliably distinguishing tumor images from normal ones.

| Metric   | Validation Set | Test Set |
| -------- | -------------- | -------- |
| Accuracy | ~91%           | ~89%     |
| F1 Score | ~0.91          | ~0.88    |

## Repository Contents

- Python notebooks containing all experimental code.  
- Saved model files for quick loading and inference.  
- Dataset folders with original and augmented images.

To load a trained model from disk, use:

```python
from tensorflow.keras.models import load_model
model = load_model('path_to_model_file.model')
