# ✍️ Handwritten Digits Recognition


A Machine Learning and Neural Network project for recognizing handwritten
digits from **0 to 9** using the **MNIST dataset**.

The project performs complete data analysis, image and pixel-level analysis,
data preprocessing, multiple machine learning model comparisons, neural
network classification, error analysis, and custom handwritten digit
prediction.

---

## 📌 Project Overview

Handwritten digit recognition is a fundamental Computer Vision and
classification problem.

The objective of this project is to build a system that can recognize an
image of a handwritten digit and classify it into one of ten classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The project follows a complete Machine Learning workflow:

```text
Dataset Loading
       ↓
Data Understanding
       ↓
Exploratory Data Analysis
       ↓
Image Analysis
       ↓
Pixel Analysis
       ↓
Data Preprocessing
       ↓
Train / Validation / Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Error Analysis
       ↓
Custom Handwritten Digit Prediction
       ↓
Final Model Selection
```

---

# 🎯 Problem Statement

The main objectives of this project are:

1. Perform a complete data analysis of the handwritten digit dataset.
2. Classify a handwritten digit image into one of 10 classes representing
   integer values from 0 to 9.
3. Compare multiple classification models and identify the model that
   performs best.
4. Analyze model errors and identify commonly confused digit classes.
5. Select the most suitable model for production use.

---

# 📊 Dataset

The project uses the **MNIST (Modified National Institute of Standards and
Technology)** handwritten digit dataset.

The dataset contains grayscale images of handwritten digits from 0 to 9.

The MNIST dataset was loaded directly using **TensorFlow/Keras**:

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---

## Dataset Characteristics

| Property | Value |
|---|---:|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Height | 28 pixels |
| Image Width | 28 pixels |
| Pixels per Image | 784 |
| Image Type | Grayscale |
| Number of Classes | 10 |
| Classes | 0–9 |
| Original Pixel Range | 0–255 |

---

# 🔎 Exploratory Data Analysis

Several analyses were performed to understand the dataset before model
training.

## Dataset Shape

```text
Training images : (60000, 28, 28)
Training labels : (60000,)

Testing images  : (10000, 28, 28)
Testing labels  : (10000,)
```

## Pixel Statistics

| Statistic | Value |
|---|---:|
| Minimum Pixel Value | 0 |
| Maximum Pixel Value | 255 |
| Mean Pixel Value | 33.3184 |
| Standard Deviation | 78.5675 |
| Median Pixel Value | 0 |

## Background vs Non-Background Pixels

Approximately:

```text
Background pixels     : 80.88%
Non-background pixels : 19.12%
```

This shows that most pixels in an MNIST image belong to the background.

---

# 📈 Class Distribution

The dataset contains ten classes representing digits from 0 to 9.

Training-set distribution:

| Digit | Count | Percentage |
|---:|---:|---:|
| 0 | 5,923 | 9.87% |
| 1 | 6,742 | 11.24% |
| 2 | 5,958 | 9.93% |
| 3 | 6,131 | 10.22% |
| 4 | 5,842 | 9.74% |
| 5 | 5,421 | 9.04% |
| 6 | 5,918 | 9.86% |
| 7 | 6,265 | 10.44% |
| 8 | 5,851 | 9.75% |
| 9 | 5,949 | 9.92% |

The classes are reasonably balanced, although the number of samples varies
slightly between digits.

---

# 🧹 Data Quality Checks

The following checks were performed.

## Missing Values

```text
NaN values in X_train : 0
NaN values in X_test  : 0
NaN values in y_train : 0
NaN values in y_test  : 0
```

## Duplicate Images

```text
Total training images : 60,000
Unique training images: 60,000
Duplicate images      : 0
```

Therefore, no duplicate training images were detected.

---

# 🖼️ Image Analysis

Each MNIST image contains:

```text
Height       : 28 pixels
Width        : 28 pixels
Total Pixels : 784
Channels     : Grayscale
```

The original images were visualized to understand handwritten digit
patterns and variations between samples.

---

# 🔢 Pixel Analysis

The original pixel values ranged from:

```text
Minimum : 0
Maximum : 255
```

For machine learning and neural network training, pixel values were
normalized to the range:

```text
0.0 → 1.0
```

Normalization:

```python
X_train_normalized = X_train.astype("float32") / 255.0
X_test_normalized = X_test.astype("float32") / 255.0
```

---

# 🔄 Feature Preparation

For traditional machine learning models, each 28 × 28 image was flattened
into a 784-dimensional feature vector.

```text
Original image:
28 × 28

       ↓

Flatten

       ↓

784 features
```

Final feature shapes:

```text
Training Features : (60000, 784)
Testing Features  : (10000, 784)
```

---

# ✂️ Train / Validation / Test Split

The training dataset was divided into training and validation sets.

```text
Training data   : 54,000 images
Validation data :  6,000 images
Test data       : 10,000 images
```

The test dataset was kept separate for final model evaluation.

---

# 🤖 Models Evaluated

Five classification approaches were implemented:

1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Machine (SVM) with RBF Kernel**
3. **Logistic Regression**
4. **Random Forest**
5. **Neural Network using TensorFlow/Keras**

---

# 1️⃣ K-Nearest Neighbors

Different values of K were evaluated using the validation dataset.

| K | Validation Accuracy |
|---:|---:|
| 3 | **97.20%** |
| 5 | 96.50% |
| 7 | 96.40% |
| 9 | 96.35% |

Best value:

```text
Best K = 3
Validation Accuracy = 97.20%
```

Final test performance:

```text
Accuracy  : 96.99%
Precision : 97.01%
Recall    : 96.99%
F1 Score  : 96.99%
```

---

# 2️⃣ Support Vector Machine

An SVM using the RBF kernel was trained for handwritten digit classification.

Final test performance:

```text
Accuracy  : 97.19%
Precision : 97.19%
Recall    : 97.19%
F1 Score  : 97.19%
```

Training time:

```text
Approximately 29.99 seconds
```

Prediction time:

```text
Approximately 33.26 seconds
```

---

# 3️⃣ Logistic Regression

Logistic Regression was implemented as a linear classification baseline.

Final test performance:

```text
Accuracy  : 92.52%
Precision : 92.51%
Recall    : 92.52%
F1 Score  : 92.51%
```

Although Logistic Regression provided a useful baseline, it performed
considerably worse than the other evaluated models.

---

# 4️⃣ Random Forest

A Random Forest classifier was trained using the 784 pixel features.

Final test performance:

```text
Accuracy  : 96.96%
Precision : 96.96%
Recall    : 96.96%
F1 Score  : 96.96%
```

Training time:

```text
Approximately 14.10 seconds
```

The model used all 784 image features and the total feature importance was
1.0.

---

# 5️⃣ Neural Network

A fully connected Neural Network was implemented using TensorFlow/Keras.

## Architecture

```text
Input Image
    ↓
Flatten
    ↓
Dense Layer — 128 neurons, ReLU
    ↓
Dense Layer — 64 neurons, ReLU
    ↓
Output Layer — 10 neurons, Softmax
```

### Model Parameters

```text
Total Parameters       : 109,386
Trainable Parameters   : 109,386
Non-trainable          : 0
```

The network was trained for 10 epochs.

Final test performance:

```text
Test Loss     : 0.0880
Test Accuracy : 97.46%
```

Classification metrics:

```text
Accuracy  : 97.46%
Precision : 97.49%
Recall    : 97.46%
F1 Score  : 97.46%
```

---

# 🏆 Model Comparison

The final test performance of all models:

| Rank | Model | Accuracy | Precision | Recall | F1 Score |
|---:|---|---:|---:|---:|---:|
| 🥇 1 | **Neural Network** | **97.46%** | **97.49%** | **97.46%** | **97.46%** |
| 🥈 2 | SVM (RBF) | 97.19% | 97.19% | 97.19% | 97.19% |
| 🥉 3 | KNN | 96.99% | 97.01% | 96.99% | 96.99% |
| 4 | Random Forest | 96.96% | 96.96% | 96.96% | 96.96% |
| 5 | Logistic Regression | 92.52% | 92.51% | 92.52% | 92.51% |

---

# 🥇 Best Model

The **Neural Network** was selected as the final model.

Performance:

```text
Accuracy  : 97.46%
Precision : 97.49%
Recall    : 97.46%
F1 Score  : 97.46%
```

The Neural Network outperformed the second-best model, SVM, by:

```text
97.46% - 97.19% = 0.27 percentage points
```

---

# 💡 Why Neural Network Was Selected

The Neural Network was selected because:

- It achieved the highest test accuracy.
- It achieved the highest F1 score.
- It performed strongly across the ten digit classes.
- It provides probability scores for all ten classes.
- It successfully predicted a custom handwritten digit.
- It provides a suitable foundation for future image-classification
  improvements.

The model produces both:

```text
Predicted Digit
       +
Prediction Confidence
```

Example:

```text
Predicted Digit : 5
Confidence      : 99.66%
```

---

# 🔍 Error Analysis

The selected Neural Network was further analyzed to understand its
classification errors.

## Overall Results

```text
Total Test Images     : 10,000
Correct Predictions   : 9,746
Incorrect Predictions : 254
Accuracy              : 97.46%
Error Rate            : 2.54%
```

---

# 📊 Per-Digit Error Analysis

| Digit | Total | Correct | Incorrect | Accuracy | Error Rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 980 | 970 | 10 | 98.98% | 1.02% |
| 1 | 1,135 | 1,112 | 23 | 97.97% | 2.03% |
| 2 | 1,032 | 1,010 | 22 | 97.87% | 2.13% |
| 3 | 1,010 | 994 | 16 | 98.42% | 1.58% |
| 4 | 982 | 956 | 26 | 97.35% | 2.65% |
| 5 | 892 | 865 | 27 | 96.97% | 3.03% |
| 6 | 958 | 933 | 25 | 97.39% | 2.61% |
| 7 | 1,028 | 1,003 | 25 | 97.57% | 2.43% |
| 8 | 974 | 952 | 22 | 97.74% | 2.26% |
| 9 | 1,009 | 951 | 58 | 94.25% | 5.75% |

### Easiest Digit

```text
Digit 0
Accuracy   : 98.98%
Error Rate : 1.02%
```

### Most Difficult Digit

```text
Digit 9
Accuracy   : 94.25%
Error Rate : 5.75%
```

---

# 🔀 Common Misclassification Patterns

The most frequent classification errors included:

| Actual Digit | Predicted Digit | Number of Errors |
|---:|---:|---:|
| 9 | 7 | 18 |
| 5 | 3 | 14 |
| 9 | 8 | 13 |
| 4 | 9 | 12 |
| 9 | 4 | 11 |
| 1 | 8 | 10 |
| 8 | 3 | 9 |
| 7 | 3 | 9 |
| 6 | 0 | 8 |
| 2 | 8 | 7 |

The results show that digit **9** is particularly difficult for the model
and is commonly confused with digits such as 7, 8, and 4.

---

# 🎯 Prediction Confidence Analysis

The Neural Network's prediction confidence was also analyzed.

```text
Average Prediction Confidence
: 98.44%

Minimum Prediction Confidence
: 24.45%

Maximum Prediction Confidence
: 100%
```

For incorrectly classified images:

```text
Average Confidence : 77.72%
Minimum Confidence : 24.45%
Maximum Confidence : 99.9973%
```

An important observation is that the model can occasionally make a
high-confidence incorrect prediction.

Example:

```text
Actual     : 6
Predicted  : 4
Confidence : 99.9973%
```

This demonstrates that a high probability score should not always be
interpreted as absolute certainty.

---

# ✍️ Custom Handwritten Digit Prediction

The trained Neural Network was tested using a new handwritten digit image
outside the original MNIST test dataset.

The image was processed using:

```text
Input Image
    ↓
Grayscale Conversion
    ↓
Resize to 28 × 28
    ↓
Pixel Normalization
    ↓
Image Inversion if Required
    ↓
Neural Network
    ↓
Prediction
```

## Custom Prediction Result

```text
Predicted Digit : 5
Confidence      : 99.66%
```

This demonstrates that the trained model can be applied to previously unseen
handwritten digit images after appropriate preprocessing.

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Machine Learning

- Scikit-learn
- K-Nearest Neighbors
- Support Vector Machine
- Logistic Regression
- Random Forest

## Deep Learning

- TensorFlow
- Keras
- Neural Networks

## Data Analysis

- NumPy
- Pandas

## Visualization

- Matplotlib
- Seaborn

## Image Processing

- Pillow

## Development Environment

- Jupyter Notebook
- VS Code

---

# 📁 Project Structure

```text
Handwritten-Digits-Recognition/
│
├── HandwrittenDigits.ipynb
├── mnist_digit_classifier.keras
├── Challenges_and_Solutions.txt
├── README.md
│
└── images/
    ├── MNIST_0.webp
    ├── MNIST_1.webp
    ├── MNIST_2.jpg
    ├── MNIST_3.webp
    ├── MNIST_4.png
    ├── MNIST_5.webp
    ├── MNIST_6.png
    ├── MNIST_7.webp
    ├── MNIST_8.jpg
    └── MNIST_9.jpg

---

# 💾 Saved Model

The final Neural Network model was saved in Keras format:

```text
mnist_digit_classifier.keras
```

The model can be loaded using:

```python
from tensorflow.keras.models import load_model

model = load_model("mnist_digit_classifier.keras")
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/DEVAM0711/Handwritten-Digits-Recognition.git
```

## 2. Navigate to the Project Folder

```bash
cd Handwritten-Digits-Recognition
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---


## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
PRCP-1002-HandwrittenDigits.ipynb
```

Run the notebook from the first cell to the last cell.

---

# 📌 Key Results

```text
Dataset
-------
Training Images : 60,000
Testing Images  : 10,000
Classes         : 10
Image Size      : 28 × 28

Best Model
----------
Neural Network

Test Accuracy
-------------
97.46%

Precision
---------
97.49%

Recall
------
97.46%

F1 Score
--------
97.46%

Test Errors
-----------
254

Error Rate
----------
2.54%

Custom Prediction
------------------
Digit       : 5
Confidence  : 99.66%
```

---

# 🔮 Future Improvements

Although the current Neural Network achieved strong performance, several
improvements can be explored.

### 1. Convolutional Neural Network

A CNN could be implemented because convolutional layers are specifically
designed to learn spatial patterns in images.

### 2. Data Augmentation

Techniques such as:

- Rotation
- Shifting
- Zooming
- Shearing

could be used to improve robustness to different handwriting styles.

### 3. Hyperparameter Tuning

The Neural Network could be further optimized by experimenting with:

- Number of layers
- Number of neurons
- Learning rate
- Batch size
- Number of epochs
- Dropout
- Optimizers

### 4. Improved Image Preprocessing

Custom handwritten images could be improved using:

- Cropping
- Centering
- Thresholding
- Noise removal
- Stroke normalization

### 5. Deployment

The trained model could be deployed using:

- Streamlit
- FastAPI
- Flask
- REST API

to create an interactive handwritten digit recognition application.

---

# 📚 Learning Outcomes

Through this project, the following skills were developed:

- Understanding image classification problems
- Working with the MNIST dataset
- Exploratory Data Analysis
- Image and pixel-level analysis
- Data preprocessing
- Feature scaling
- Train/validation/test splitting
- Classification using Scikit-learn
- Neural Network development using TensorFlow/Keras
- Model evaluation
- Classification reports
- Error analysis
- Prediction confidence analysis
- Model comparison
- Model selection
- Custom image prediction
- Machine Learning project documentation

---

# 👨‍💻 Author

**Devam Jasani**

Data Science & AI/ML Engineer



---

# 👨‍💻Contributing 



* Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

---

## ⭐ If you found this project useful, don't forget to star this repository!
