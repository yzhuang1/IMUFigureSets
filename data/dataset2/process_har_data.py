#!/usr/bin/env python3
"""
Process Human Activity Recognition dataset to create X.npy and y.npy files
similar to the structure in dataset1
"""
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

def load_har_data():
    """Load and process Human Activity Recognition dataset"""
    dataset_path = Path(".")  # Current directory (dataset2)

    # Load training data
    X_train = pd.read_csv(dataset_path / "train" / "X_train.txt",
                         delim_whitespace=True, header=None)
    y_train = pd.read_csv(dataset_path / "train" / "y_train.txt",
                         header=None, names=['activity'])
    subject_train = pd.read_csv(dataset_path / "train" / "subject_train.txt",
                               header=None, names=['subject'])

    # Load test data
    X_test = pd.read_csv(dataset_path / "test" / "X_test.txt",
                        delim_whitespace=True, header=None)
    y_test = pd.read_csv(dataset_path / "test" / "y_test.txt",
                        header=None, names=['activity'])
    subject_test = pd.read_csv(dataset_path / "test" / "subject_test.txt",
                              header=None, names=['subject'])

    # Load feature names
    features = pd.read_csv(dataset_path / "features.txt",
                          delim_whitespace=True, header=None,
                          names=['id', 'feature'])

    # Load activity labels
    activity_labels = pd.read_csv(dataset_path / "activity_labels.txt",
                                 delim_whitespace=True, header=None,
                                 names=['id', 'activity'])

    # Combine training and test data
    X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    subjects_combined = pd.concat([subject_train, subject_test], axis=0, ignore_index=True)

    # Convert to numpy arrays
    X = X_combined.values.astype(np.float32)
    y = (y_combined['activity'].values - 1).astype(np.int64)  # Convert to 0-based indexing

    # Save as numpy arrays (similar to dataset1)
    np.save(dataset_path / "X.npy", X)
    np.save(dataset_path / "y.npy", y)

    # Create metadata file
    metadata = {
        "dataset_name": "Human Activity Recognition Using Smartphones",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(activity_labels),
        "feature_names": features['feature'].tolist(),
        "class_names": activity_labels['activity'].tolist(),
        "data_shape": X.shape,
        "label_shape": y.shape,
        "data_type": "sensor_data",
        "description": "Human Activity Recognition database built from recordings of 30 subjects performing activities of daily living while carrying a waist-mounted smartphone with embedded inertial sensors."
    }

    with open(dataset_path / "har_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create a sample CSV file (similar to dataset1)
    sample_size = min(1000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    sample_data = pd.DataFrame(X[indices])
    sample_data['activity'] = y[indices]
    sample_data.to_csv(dataset_path / "har_sample.csv", index=False)

    print(f"Processed HAR dataset:")
    print(f"- Shape: {X.shape}")
    print(f"- Classes: {len(activity_labels)}")
    print(f"- Saved X.npy, y.npy, har_metadata.json, and har_sample.csv")

    return X, y, metadata

if __name__ == "__main__":
    X, y, metadata = load_har_data()