# Brain Tumor Detection Using CNN

This project involves developing a brain tumor detection model utilizing a convolutional neural network built with TensorFlow and Keras. The model is trained on brain MRI images sourced from Kaggle, available here: [Brain MRI Images Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

## Dataset Details

The dataset consists of two main folders:  
- **yes**: Contains 155 MRI scans showing presence of brain tumors  
- **no**: Contains 98 MRI scans without tumors  

In total, there are 253 MRI images across both categories.

---

## Getting Started

> **Note:** GitHub’s default viewer might not properly render Jupyter notebooks. For a better experience, consider using [nbviewer](https://nbviewer.jupyter.org/) to open the notebooks.

---

## Data Augmentation

**Why augment data?**  
The dataset is relatively small and imbalanced, with fewer samples for non-tumorous cases. To address this and help the model generalize better, data augmentation techniques were applied.

- Original dataset: 155 tumor-positive and 98 tumor-negative images (253 total)  
- After augmentation: 1085 tumor-positive and 980 tumor-negative images (2065 total), including the original images  

The augmented images are saved in the folder named `augmented data`.

More details can be found in the "Data Augmentation" notebook.

---

## Preprocessing Steps

Each MRI image undergoes the following preprocessing before being fed into the network:

1. Crop the image to focus only on the brain region, removing unnecessary background.  
2. Resize all images to a uniform shape of (240, 240, 3) to ensure consistency.  
3. Normalize pixel intensities to values between 0 and 1 to facilitate better training performance.

---

## Dataset Splitting

The dataset was partitioned as follows:

- 70% for training  
- 15% for validation  
- 15% for testing  

---

## Neural Network Design

The network accepts input images of shape (240, 240, 3) and processes them through these layers:

1. Zero Padding layer with pool size (2, 2)  
2. Convolutional layer with 32 filters, each of size (7, 7), stride 1  
3. Batch Normalization layer to accelerate and stabilize training  
4. ReLU activation layer  
5. Max Pooling layer with filter size 4 and stride 4  
6. Another Max Pooling layer with filter size 4 and stride 4  
7. Flatten layer to convert 3D features into 1D vector  
8. Dense output layer with one neuron and sigmoid activation for binary classification  

### Why This Architecture?

Initial experiments using transfer learning models like ResNet50 and VGG16 resulted in overfitting, primarily due to the limited dataset size and the hardware constraints (training was conducted on a 6th gen Intel i7 CPU with 8GB RAM). Given these factors, a simpler CNN architecture was designed and trained from scratch, achieving effective results with reasonable computational demands.

---

## Model Training

The model was trained for 24 epochs. The training process showed steady improvement in both loss and accuracy, with the highest validation accuracy reached on the 23rd epoch.

---

## Evaluation and Results

The best performing model achieved the following on the test set:

- **Accuracy:** 88.7%  
- **F1 Score:** 0.88  

Considering the relatively balanced nature of the dataset, these metrics demonstrate strong model performance.

| Metric   | Validation Set | Test Set |
| -------- | -------------- | -------- |
| Accuracy | 91%            | 89%      |
| F1 Score | 0.91           | 0.88     |

---

## Additional Information

What’s included in this repository?

- Jupyter notebooks containing all code and experiments.  
- Model weights saved as `.model` files, with the best model named `cnn-parameters-improvement-23-0.91.model`.  
- Original dataset in the `yes` and `no` folders.  
- Augmented dataset in the `augmented data` folder.

You can load the best saved model using the following code snippet:

```python
from tensorflow.keras.models import load_model

best_model = load_model('models/cnn-parameters-improvement-23-0.91.model')
