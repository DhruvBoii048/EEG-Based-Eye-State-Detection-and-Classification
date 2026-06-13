"""
Dataset Preprocessing Pipeline for EEG Eye State Classification

Pipeline:
1. Data Cleaning
2. Feature Engineering
3. Feature Selection using XGBoost
4. Min-Max Normalization

Input:
    Dataset/EEG Eye State.arff

Generated Outputs:
    EEG_Eye_State_Feature_Engineered.arff
    EEG_Eye_State_Selected_Features.arff
    EEG_Eye_State_Normalized_MinMax.arff
"""

import pandas as pd
import numpy as np

from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = "EEG Eye State.arff"

FEATURE_ENGINEERED_FILE = "EEG_Eye_State_Feature_Engineered.arff"
SELECTED_FEATURES_FILE = "EEG_Eye_State_Selected_Features.arff"
NORMALIZED_FILE = "EEG_Eye_State_Normalized_MinMax.arff"


# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------

def save_arff(df, filename, relation="EEG_Eye_State"):
    """
    Save DataFrame as ARFF file.
    """

    with open(filename, "w") as f:

        f.write(f"@RELATION {relation}\n\n")

        for column in df.columns:

            if column == "eyeDetection":
                f.write("@ATTRIBUTE eyeDetection {0,1}\n")

            else:
                f.write(f"@ATTRIBUTE {column} NUMERIC\n")

        f.write("\n@DATA\n")

        for _, row in df.iterrows():
            f.write(",".join(map(str, row.values)) + "\n")


# --------------------------------------------------
# STEP 1: DATA CLEANING
# --------------------------------------------------

def clean_dataset(df):

    original_rows = len(df)

    print(f"Original records: {original_rows}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing numeric values
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())

    print(f"Records after cleaning: {len(df)}")

    return df


# --------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# --------------------------------------------------

def feature_engineering(df):

    eeg_channels = [
        "AF3", "F7", "F3", "FC5", "T7", "P7",
        "O1", "O2", "P8", "T8", "FC6",
        "F4", "F8", "AF4"
    ]

    # Statistical Features
    df["mean_EEG"] = df[eeg_channels].mean(axis=1)
    df["std_EEG"] = df[eeg_channels].std(axis=1)
    df["median_EEG"] = df[eeg_channels].median(axis=1)

    # Channel Difference Features
    df["F3_F4_diff"] = df["F3"] - df["F4"]
    df["AF3_AF4_diff"] = df["AF3"] - df["AF4"]

    # Aggregate Feature
    df["total_EEG"] = df[eeg_channels].sum(axis=1)

    # Lag Features
    for channel in eeg_channels:
        df[f"{channel}_lag1"] = df[channel].shift(1)

    df = df.fillna(0)

    return df


# --------------------------------------------------
# STEP 3: FEATURE SELECTION
# --------------------------------------------------

def feature_selection(df):

    target_column = "eyeDetection"

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    selected_features = importance_df[
        importance_df["Importance"] > 0.01
    ]["Feature"].tolist()

    print("\nSelected Features:")
    print(selected_features)

    selected_df = df[selected_features + [target_column]]

    return selected_df


# --------------------------------------------------
# STEP 4: NORMALIZATION
# --------------------------------------------------

def normalize_dataset(df):

    scaler = MinMaxScaler()

    target = df["eyeDetection"]

    features = df.drop("eyeDetection", axis=1)

    normalized_features = scaler.fit_transform(features)

    normalized_df = pd.DataFrame(
        normalized_features,
        columns=features.columns
    )

    normalized_df["eyeDetection"] = target.values

    return normalized_df


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    print("Loading dataset...")

    data, meta = arff.loadarff(INPUT_FILE)

    df = pd.DataFrame(data)

    # Convert target column from bytes to integer
    if df["eyeDetection"].dtype == object:
        df["eyeDetection"] = df["eyeDetection"].apply(
            lambda x: int(x.decode("utf-8"))
        )

    print("Cleaning dataset...")
    df = clean_dataset(df)

    print("Generating engineered features...")
    engineered_df = feature_engineering(df)

    save_arff(
        engineered_df,
        FEATURE_ENGINEERED_FILE,
        "EEG_Eye_State_Feature_Engineered"
    )

    print("Performing feature selection...")
    selected_df = feature_selection(engineered_df)

    save_arff(
        selected_df,
        SELECTED_FEATURES_FILE,
        "EEG_Eye_State_Selected_Features"
    )

    print("Applying Min-Max normalization...")
    normalized_df = normalize_dataset(selected_df)

    save_arff(
        normalized_df,
        NORMALIZED_FILE,
        "EEG_Eye_State_Normalized_MinMax"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()