# EEG Eye State Classification using Machine Learning

## Overview

This project focuses on classifying human eye states (Open / Closed) using Electroencephalography (EEG) signals collected from multiple EEG channels.

The objective is to explore machine learning techniques for brain-signal-based classification and compare their performance on EEG data. The project includes exploratory data analysis, signal visualization, preprocessing, model training, and evaluation.

---

## Features

* EEG signal analysis and visualization
* Correlation analysis between EEG channels
* Time-series visualization of EEG recordings
* Spectrogram generation for frequency-domain analysis
* Data preprocessing and normalization pipeline
* Machine learning model training and evaluation
* Performance comparison between multiple classifiers
* Model interpretability using SHAP values

---

## Dataset

The project uses the **EEG Eye State Dataset**, which contains EEG recordings captured from 14 electrode channels along with eye-state labels.

### EEG Channels

* AF3
* F7
* F3
* FC5
* T7
* P7
* O1
* O2
* P8
* T8
* FC6
* F4
* F8
* AF4

### Target Variable

| Value | Meaning    |
| ----- | ---------- |
| 0     | Eye Open   |
| 1     | Eye Closed |

Dataset file:

```text
Dataset/EEG Eye State.arff
```

---

## Project Structure

```text
EEG-Based-Eye-State-Detection-and-Classification/
│
├── Dataset/
│   ├── EEG Eye State.arff
│   └── datasetcleaning.py
│
├── eyestate_Final.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Exploratory Data Analysis

The project includes several visualization techniques:

* Correlation Heatmap
* EEG Time-Series Visualization
* EEG Signal Subplots
* Spectrogram Analysis
* Eye-State Distribution Pie Chart
* Boxplot Analysis

These visualizations help understand signal behavior and feature relationships before model training.

---

## Machine Learning Models

### XGBoost

Gradient boosting-based ensemble learning model used for binary classification.

**Evaluation Metrics**

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### K-Nearest Neighbors (KNN)

Distance-based classification model with hyperparameter tuning using GridSearchCV.

**Optimized Parameters**

* Number of Neighbors
* Distance Metric
* Weight Function

---

## Data Preprocessing

The preprocessing pipeline includes:

* Missing value inspection
* Duplicate record removal
* Feature scaling
* Dataset cleaning
* Feature engineering experiments
* Feature selection experiments

The preprocessing utilities are available in:

```text
Dataset/datasetcleaning.py
```

---

## Model Interpretability

SHAP (SHapley Additive exPlanations) is used to analyze feature importance and understand model decision-making behavior.

This provides insight into which EEG channels contribute most significantly to eye-state prediction.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/DhruvBoii048/EEG-Based-Eye-State-Detection-and-Classification.git
cd EyeStateClassification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main classification pipeline:

```bash
python eyestate_Final.py
```

The script performs:

1. Dataset loading
2. Data visualization
3. Preprocessing
4. Model training
5. Evaluation
6. SHAP-based interpretability analysis

---

## Results

The project evaluates machine learning models using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* RMSE
* R² Score

Performance comparison helps identify the most suitable model for EEG eye-state classification.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SciPy
* SHAP

---

## Authors

**Dhruv Maheshwari**

**Ritwik Mittal**

Hackathon Project

---

## Future Improvements

* Deep Learning based EEG classification
* Real-time EEG stream processing
* Additional feature extraction techniques
* Advanced signal processing methods
* Deployment as a web application
