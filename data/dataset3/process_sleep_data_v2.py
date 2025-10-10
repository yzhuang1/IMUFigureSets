#!/usr/bin/env python3
"""
Process ISRUC-SLEEP dataset to create X.npy and y.npy files
Handles both A2/A1 and M2/M1 reference electrode naming conventions

Dataset3: Sleep Stage Classification
- Input: 6 EEG channels (F3, C3, O1, F4, C4, O2)
- Output: Sleep stages from 1_1.txt (Expert 1 annotations)
- Format: 30-second epochs at 200 Hz (6000 samples per epoch)
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
import pyedflib

def find_eeg_channels(signal_labels):
    """
    Find the 6 EEG channels regardless of reference naming (A1/A2, M1/M2, or no reference)

    Returns:
        list of (index, channel_name) tuples, or None if not all 6 found
    """
    # Target channel positions (F3, C3, O1, F4, C4, O2)
    target_positions = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']

    found_channels = []
    for pos in target_positions:
        # Look for this position with any reference (A1, A2, M1, M2) OR exact match
        for i, label in enumerate(signal_labels):
            # Match with reference: F3-A2, F3-M2, etc.
            if label.startswith(pos + '-'):
                found_channels.append((i, label))
                break
            # Match without reference (e.g., subject 040): F3, C3, etc.
            elif label == pos:
                found_channels.append((i, label))
                break
        else:
            # Position not found
            return None

    return found_channels


def process_single_subject(subject_dir):
    """
    Process a single subject's data

    Args:
        subject_dir: Path to subject directory (e.g., .../001/1/)

    Returns:
        epochs: numpy array of shape (n_epochs, n_channels, samples_per_epoch)
        labels: numpy array of shape (n_epochs,)
    """
    # Find the .rec file
    rec_files = list(subject_dir.glob("*.rec"))
    if len(rec_files) == 0:
        return None, None
    rec_file = rec_files[0]

    # Find the annotation file (using expert 1: *_1.txt)
    annotation_files = list(subject_dir.glob("*_1.txt"))
    if len(annotation_files) == 0:
        return None, None
    annotation_file = annotation_files[0]

    # Read EDF file
    try:
        f = pyedflib.EdfReader(str(rec_file))
        signal_labels = f.getSignalLabels()

        # Find the 6 EEG channels
        channel_info = find_eeg_channels(signal_labels)
        if channel_info is None:
            f.close()
            return None, None

        channel_indices = [idx for idx, _ in channel_info]
        channel_names = [name for _, name in channel_info]

        # Read all 6 EEG channels
        fs = int(f.getSampleFrequency(channel_indices[0]))  # Should be 200 Hz
        signals = []
        for idx in channel_indices:
            signal = f.readSignal(idx)
            signals.append(signal)

        f.close()

        # Stack into (n_channels, n_samples)
        signals = np.array(signals, dtype=np.float32)

        # Segment into 30-second epochs
        epoch_duration = 30  # seconds
        samples_per_epoch = fs * epoch_duration  # 200 * 30 = 6000
        n_samples = signals.shape[1]
        n_epochs = n_samples // samples_per_epoch

        # Truncate to complete epochs and reshape
        signals_truncated = signals[:, :n_epochs * samples_per_epoch]
        # Reshape to (n_channels, n_epochs, samples_per_epoch) then transpose
        epochs = signals_truncated.reshape(len(channel_indices), n_epochs, samples_per_epoch)
        # Transpose to (n_epochs, n_channels, samples_per_epoch)
        epochs = epochs.transpose(1, 0, 2)

    except Exception as e:
        print(f"Error reading {rec_file}: {e}")
        return None, None

    # Read annotations
    try:
        with open(annotation_file, 'r') as f:
            labels = [int(line.strip()) for line in f if line.strip()]  # Skip empty lines
        labels = np.array(labels, dtype=np.int64)

        # Ensure labels match epochs
        min_len = min(len(labels), len(epochs))
        epochs = epochs[:min_len]
        labels = labels[:min_len]

    except Exception as e:
        print(f"Error reading {annotation_file}: {e}")
        return None, None

    return epochs, labels, channel_names


def process_sleep_dataset():
    """Process the entire ISRUC-SLEEP dataset"""

    dataset_path = Path("ISRUC_Subgroup1_extracted")

    # Get all subject directories
    subject_dirs = []
    for subject_folder in sorted(dataset_path.glob("*")):
        if subject_folder.is_dir():
            # Find the first subdirectory that contains .rec files
            for subdir in sorted(subject_folder.glob("*")):
                if subdir.is_dir() and list(subdir.glob("*.rec")):
                    subject_dirs.append(subdir)
                    break

    print(f"Found {len(subject_dirs)} subjects")
    print(f"Processing 6 EEG channels: F3, C3, O1, F4, C4, O2")
    print(f"  (Accepts both A1/A2 and M1/M2 reference naming)")
    print(f"Using Expert 1 annotations (*_1.txt)")
    print()

    all_epochs = []
    all_labels = []
    subject_info = []

    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        result = process_single_subject(subject_dir)

        if result[0] is not None and result[1] is not None:
            epochs, labels, channel_names = result
            all_epochs.append(epochs)
            all_labels.append(labels)
            subject_info.append({
                'subject_id': subject_dir.parent.name,
                'n_epochs': len(epochs),
                'epochs_shape': epochs.shape,
                'channels': channel_names
            })
        else:
            print(f"Skipped {subject_dir.parent.name}")

    # Concatenate all subjects
    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"\nFinal dataset shape:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  X dtype: {X.dtype}")
    print(f"  y dtype: {y.dtype}")

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM'}
    for stage, count in zip(unique, counts):
        stage_name = stage_names.get(stage, f'Unknown({stage})')
        print(f"  {stage_name:10s} (label {stage}): {count:6d} epochs ({count/len(y)*100:5.2f}%)")

    # Save as numpy arrays (consistent with dataset1 and dataset2)
    np.save("X.npy", X)
    np.save("y.npy", y)
    print(f"\nSaved X.npy and y.npy")

    # Get canonical channel names from first subject
    canonical_channels = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']

    # Create metadata file
    metadata = {
        "dataset_name": "ISRUC-SLEEP Subgroup 1",
        "n_samples": len(X),
        "n_subjects": len(subject_info),
        "n_channels": X.shape[1],
        "samples_per_epoch": X.shape[2],
        "n_classes": len(unique),
        "channel_names": canonical_channels,
        "class_names": [stage_names.get(s, f'Unknown({s})') for s in unique],
        "class_labels": unique.tolist(),
        "data_shape": list(X.shape),
        "label_shape": list(y.shape),
        "data_type": "sleep_eeg",
        "sampling_rate": 200,
        "epoch_duration": 30,
        "annotation_source": "Expert 1 (*_1.txt)",
        "description": "ISRUC-SLEEP dataset with 6 EEG channels for sleep stage classification. Each sample is a 30-second epoch with 6 channels sampled at 200 Hz. Accepts both A1/A2 and M1/M2 reference electrode naming.",
        "subjects": subject_info
    }

    with open("sleep_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved sleep_metadata.json")

    # Create a sample CSV file (first 100 epochs, flattened features)
    sample_size = min(100, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)

    # Flatten the 3D data for CSV (n_samples, n_channels * samples_per_epoch)
    X_flat = X.reshape(X.shape[0], -1)
    sample_data = pd.DataFrame(X_flat[sample_indices])
    sample_data['sleep_stage'] = y[sample_indices]
    sample_data.to_csv("sleep_sample.csv", index=False)
    print(f"Saved sleep_sample.csv (100 sample epochs)")

    return X, y, metadata


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    X, y, metadata = process_sleep_dataset()
