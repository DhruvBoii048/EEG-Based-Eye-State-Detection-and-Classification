"""
Eye State Classification Using EEG Signals

Authors:
Dhruv Maheshwari
Ritwik Mittal

Hackathon Project

Description:
Classification of eye-open and eye-closed states
using EEG recordings and machine learning models.

Models:
- XGBoost
- K-Nearest Neighbors (KNN)
"""

# ---

# IMPORTING LIBRARIES

import numpy as np
import pandas as pd
from scipy.io import arff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# ---

# LOADING AND PREVIEWING THE DATA

data, meta = arff.loadarff('Dataset/EEG Eye State.arff')
df = pd.DataFrame(data)
df.head(20)

df.tail()
df.info()
df.describe()

print(df.isnull().sum())

df.duplicated().sum()

# ---

# CORRELATION MATRIX AND HEATMAP

correlation_matrix = df.corr()

plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# ---

# VISUALIZING TIME SERIES DATA

plt.figure(figsize=(10, 6))
plt.plot(df.iloc[:1000, 0], color='blue')
plt.title("EEG Time Series (Channel 1)")
plt.xlabel("Time (Samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# ---

# SPECTOGRAM

from scipy.signal import spectrogram

data = df.iloc[:, 0].values

frequencies, times, Sxx = spectrogram(data, fs=100)

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.title("EEG Spectrogram (Channel 1)")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Intensity [dB]')
plt.show()

# ---

# ENCODING AND VISUALIZING EYE DETECTION

df['eyeDetection'] = df['eyeDetection'].apply(lambda x: int(x.decode('utf-8')))

sns.set(style="whitegrid")

plt.figure(figsize=(16, 8))

columns_to_plot = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

for column in columns_to_plot:
    plt.plot(df.index, df[column], label=column, linewidth=1.2)


plt.title('Overlay of EEG Signals (Clear Waveform Interactions)', fontsize=16, weight='bold')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('EEG Signal Value', fontsize=12)
plt.legend(loc='upper right', ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)


plt.tight_layout()
plt.show()

# ---

# SUBPLOTS OF EEG SIGNALS

columns_to_plot = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

num_rows = 4
num_cols = 4
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12))

for i, column in enumerate(columns_to_plot):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].plot(df.index, df[column], color='b', linewidth=1.2)
    axes[row, col].set_title(f'Signal: {column}', fontsize=10)
    axes[row, col].set_xlabel('Sample Index', fontsize=8)
    axes[row, col].set_ylabel('EEG Value', fontsize=8)
    axes[row, col].grid(True, linestyle='--', alpha=0.6)

    axes[row, col].set_ylim([df[column].min() - 100, df[column].max() + 100])

for i in range(len(columns_to_plot), num_rows * num_cols):
    fig.delaxes(axes.flat[i])

plt.tight_layout()

plt.show()

# ---

# PIE CHART FOR EYE DETECTION RESULTS

plt.figure(figsize = (8,5))
labels = ["close", "open"]
explode = [0, 0.1]
colors = ["#282d63", "#db0d8c"]

wedges, texts, autotexts = plt.pie(df.eyeDetection.value_counts(),
                                    labels=labels,
                                    explode=explode,
                                    colors=colors,
                                    autopct='%1.1f%%',
                                    textprops={'color': 'white'},
                                    labeldistance=1.1,
                                    wedgeprops={'linewidth': 2, 'edgecolor': 'black'})


for text in texts:
    text.set_color('black')

plt.title('Eye Detection Results')
plt.show()

# ---

# LABEL ENCODING AND DATA SPLITTING

label_encoder = LabelEncoder()
df['eyeDetection'] = label_encoder.fit_transform(df['eyeDetection'])

df.head(10)

df.info()

x=df.drop(['eyeDetection'],axis=1)
y=df['eyeDetection']

y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=120)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the electrode channel names are in the columns
plt.figure(figsize=(16, 8))

# Creating boxplots for each column in the dataframe
sns.boxplot(data=df, orient='v')

# Setting the title and labels
plt.title('Boxplot for the Dataset', fontsize=16, weight='bold')
plt.xlabel('Electrode Channels', fontsize=12)
plt.ylabel('Values', fontsize=12)

# Rotating the x-axis labels for better readability
plt.xticks(rotation=90)

# Displaying the plot
plt.show()

# ---

# XgBoost MODEL

## Initialising and declaring Variables

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 2,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 1
}

num_boost_round = 1000

## Memory Usage and Training Time while Training the model

import time

start = time.time()
xgb_model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
print(f"Training time: {time.time() - start:.2f} seconds")

## Prediction

y_pred_prob = xgb_model.predict(dtest)
y_pred_xgb = np.round(y_pred_prob)

## Evaluating the Model on Metrices

# Print accuracy
print(f'XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}')

# Print classification report
print(classification_report(y_test, y_pred_xgb))

cm = confusion_matrix(y_test, y_pred_xgb)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ---

# KNN MODEL

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
grid_search.fit(x_train, y_train)

best_knn = grid_search.best_estimator_

best_knn

knn = KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance')

## Training Time and Memory Profiling

import time

start = time.time()
knn.fit(x_train,y_train)
print(f"Training time: {time.time() - start:.2f} seconds")

knn_predicted = knn.predict(x_test)

## KNN Model Evaluation

accuracy_knn = accuracy_score(y_test,knn_predicted)
accuracy_knn

print(f"KNN Accuracy: {accuracy_knn}")
print(classification_report(y_test, knn_predicted))

cm = confusion_matrix(y_test, knn_predicted)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ---

# EVALUATION ON GIVEN METRICES FOR THE BEST MODEL

y_true_final = y_test
y_pred_final = knn_predicted
best_model = knn

## Script for Evaluating ML Models

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
import numpy as np

### Example: ML Classification Evaluation
y_true_class = y_true_final  # Ground truth labels
y_pred_class = y_pred_final  # Model predictions

## Classification Metrics
accuracy = accuracy_score(y_true_class, y_pred_class)
precision = precision_score(y_true_class, y_pred_class, average='weighted')
recall = recall_score(y_true_class, y_pred_class, average='weighted')
f1 = f1_score(y_true_class, y_pred_class, average='weighted')

## Print Classification Results
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

### Example: ML Model Evaluation
y_true_model = y_true_final  # Ground truth values
y_pred_model = y_pred_final  # Predicted values

## Model Metrics
xgboost = np.sqrt(mean_squared_error(y_true_model, y_pred_model))
r2 = r2_score(y_true_model, y_pred_model)

## Print Model Results
print(f"KNN(rms): {xgboost:.2f}, R2 Score: {r2:.2f}")

# ---

# INTERPRETABILITY

import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test)

shap_values