"""
Download and process MIT-BIH Arrhythmia Database from PhysioNet
Converts ECG data to format compatible with the ML pipeline
"""

import os
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_mitdb_data(data_dir="data", records_to_download=None):
    """
    Download MIT-BIH Arrhythmia Database records
    
    Args:
        data_dir: Directory to save data
        records_to_download: List of record numbers, or None for all records
    
    Returns:
        List of downloaded record names
    """
    print("=" * 80)
    print("DOWNLOADING MIT-BIH ARRHYTHMIA DATABASE")
    print("=" * 80)
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    mitdb_path = data_path / "mitdb"
    mitdb_path.mkdir(exist_ok=True)
    
    # MIT-BIH record numbers (48 records total)
    all_records = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
        122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
        209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
        222, 223, 228, 230, 231, 232, 233, 234
    ]
    
    if records_to_download is None:
        # Download ALL records (complete dataset)
        records_to_download = all_records
        print(f"Downloading ALL {len(records_to_download)} MIT-BIH records...")
    else:
        print(f"Downloading {len(records_to_download)} specified records...")
    
    downloaded_records = []
    
    for record_num in tqdm(records_to_download, desc="Downloading records"):
        try:
            record_name = str(record_num)
            
            # Download record if not already exists
            record_files = [
                mitdb_path / f"{record_name}.dat",
                mitdb_path / f"{record_name}.hea", 
                mitdb_path / f"{record_name}.atr"
            ]
            
            if all(f.exists() for f in record_files):
                logger.info(f"Record {record_name} already exists, skipping download")
                downloaded_records.append(record_name)
                continue
            
            # Download from PhysioNet
            logger.info(f"Downloading record {record_name}...")
            wfdb.dl_database('mitdb', str(mitdb_path), [record_name])
            
            downloaded_records.append(record_name)
            logger.info(f"Successfully downloaded record {record_name}")
            
        except Exception as e:
            logger.error(f"Failed to download record {record_num}: {e}")
            continue
    
    print(f"\n[SUCCESS] Downloaded {len(downloaded_records)} records to: {mitdb_path}")
    return downloaded_records, str(mitdb_path)

def extract_features_and_labels(record_path, record_name, segment_length=1000):
    """
    Extract features and labels from MIT-BIH record
    
    Args:
        record_path: Path to record directory
        record_name: Record name (e.g., '100')
        segment_length: Length of ECG segments in samples
    
    Returns:
        features: ECG segments (N, segment_length, 2) for 2 leads
        labels: Arrhythmia labels for each segment
        beat_labels: Beat-level annotations
    """
    try:
        # Read record and annotations
        record = wfdb.rdrecord(f"{record_path}/{record_name}")
        annotation = wfdb.rdann(f"{record_path}/{record_name}", 'atr')
        
        # Get ECG signals (usually 2 leads)
        signals = record.p_signal  # Shape: (samples, leads)
        fs = record.fs  # Sampling frequency
        
        logger.info(f"Record {record_name}: {signals.shape[0]} samples, {signals.shape[1]} leads, {fs}Hz")
        
        # Map annotation symbols to numeric labels
        # MIT-BIH annotation symbols: N (Normal), L (LBBB), R (RBBB), A (APB), V (VPB), etc.
        symbol_map = {
            'N': 0,  # Normal beat
            'L': 1,  # Left bundle branch block beat
            'R': 1,  # Right bundle branch block beat  
            'A': 2,  # Atrial premature beat
            'a': 2,  # Aberrated atrial premature beat
            'J': 2,  # Nodal (junctional) premature beat
            'S': 2,  # Supraventricular premature beat
            'V': 3,  # Premature ventricular contraction
            'E': 3,  # Ventricular escape beat
            'F': 4,  # Fusion of ventricular and normal beat
            '/': 4,  # Paced beat
            'Q': 4,  # Unclassifiable beat
            '?': 4   # Beat not classified during learning
        }
        
        # Create segments and labels
        segments = []
        segment_labels = []
        beat_annotations = []
        
        # Process in overlapping windows
        step_size = segment_length // 2  # 50% overlap
        
        for start_idx in range(0, signals.shape[0] - segment_length + 1, step_size):
            end_idx = start_idx + segment_length
            
            # Extract segment
            segment = signals[start_idx:end_idx, :]  # Shape: (segment_length, leads)
            segments.append(segment)
            
            # Find annotations within this segment
            segment_annotations = [
                (sample, symbol) for sample, symbol in zip(annotation.sample, annotation.symbol)
                if start_idx <= sample < end_idx
            ]
            
            # Determine segment label based on majority beat type
            if segment_annotations:
                symbols_in_segment = [symbol for _, symbol in segment_annotations]
                # Map symbols to numeric labels
                numeric_labels = [symbol_map.get(s, 4) for s in symbols_in_segment]  # 4 = unknown
                # Use most common label
                most_common_label = max(set(numeric_labels), key=numeric_labels.count)
                segment_labels.append(most_common_label)
                
                beat_annotations.extend([(sample - start_idx, symbol_map.get(symbol, 4)) 
                                       for sample, symbol in segment_annotations])
            else:
                # No annotations in segment, assume normal
                segment_labels.append(0)
        
        features = np.array(segments, dtype=np.float32)  # Shape: (N, segment_length, leads)
        labels = np.array(segment_labels, dtype=np.int64)
        
        logger.info(f"Extracted {len(features)} segments from record {record_name}")
        logger.info(f"Label distribution: {Counter(labels)}")
        
        return features, labels, beat_annotations
        
    except Exception as e:
        logger.error(f"Failed to process record {record_name}: {e}")
        return None, None, None

def process_mitdb_dataset(mitdb_path, output_dir="data", segment_length=1000):
    """
    Process all downloaded MIT-BIH records and create dataset
    
    Args:
        mitdb_path: Path to downloaded MIT-BIH records
        output_dir: Output directory for processed data
        segment_length: ECG segment length
    
    Returns:
        None (saves processed data to files)
    """
    print("\n" + "=" * 80)
    print("PROCESSING MIT-BIH RECORDS")
    print("=" * 80)
    
    # Find all available records
    record_files = list(Path(mitdb_path).glob("*.hea"))
    record_names = [f.stem for f in record_files]
    
    if not record_names:
        logger.error("No MIT-BIH records found!")
        return
    
    logger.info(f"Found {len(record_names)} records: {record_names}")
    
    all_features = []
    all_labels = []
    all_record_info = []
    
    # Process each record
    for record_name in tqdm(record_names, desc="Processing records"):
        features, labels, beat_annotations = extract_features_and_labels(
            mitdb_path, record_name, segment_length
        )
        
        if features is not None:
            all_features.append(features)
            all_labels.append(labels)
            
            # Store record info
            all_record_info.extend([record_name] * len(features))
            
            logger.info(f"Record {record_name}: {features.shape} features, {len(labels)} labels")
    
    if not all_features:
        logger.error("No data was successfully processed!")
        return
    
    # Combine all records
    X = np.concatenate(all_features, axis=0)  # Shape: (N, segment_length, leads)
    y = np.concatenate(all_labels, axis=0)    # Shape: (N,)
    record_info = np.array(all_record_info)
    
    print(f"\nDataset Summary:")
    print(f"  Total segments: {X.shape[0]}")
    print(f"  Segment shape: {X.shape[1:]} (length × leads)")
    print(f"  Label distribution:")
    label_names = ['Normal', 'Bundle Branch Block', 'Atrial Premature', 'Ventricular', 'Other']
    for label, name in enumerate(label_names):
        count = np.sum(y == label)
        percentage = 100 * count / len(y)
        print(f"    {label} ({name}): {count} ({percentage:.1f}%)")
    
    # Normalize features (standardize each lead independently)
    print("\nNormalizing ECG signals...")
    X_normalized = np.zeros_like(X)
    for lead in range(X.shape[2]):  # For each lead
        scaler = StandardScaler()
        # Reshape to (N*segment_length,) for fitting, then back to (N, segment_length)
        lead_data = X[:, :, lead].reshape(-1, 1)
        lead_normalized = scaler.fit_transform(lead_data)
        X_normalized[:, :, lead] = lead_normalized.reshape(X.shape[0], X.shape[1])
    
    # Save processed data in different formats for flexibility
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nSaving processed data to: {output_path}")
    
    # Format 1: NPZ file (recommended for large datasets)
    np.savez_compressed(
        output_path / "mitdb_ecg_data.npz",
        X=X_normalized,
        y=y,
        record_info=record_info,
        label_names=label_names,
        segment_length=segment_length,
        description="MIT-BIH Arrhythmia Database - ECG segments with arrhythmia labels"
    )
    print("[SUCCESS] Saved: mitdb_ecg_data.npz")
    
    # Format 2: Separate NumPy files (for pipeline compatibility)
    np.save(output_path / "X.npy", X_normalized)
    np.save(output_path / "y.npy", y)
    print("[SUCCESS] Saved: X.npy, y.npy")
    
    # Format 3: CSV file (flattened features for simple ML models)
    # Flatten ECG segments to 1D features
    X_flat = X_normalized.reshape(X_normalized.shape[0], -1)  # Shape: (N, segment_length*leads)
    
    # Create column names
    col_names = []
    for lead in range(X.shape[2]):
        for sample in range(X.shape[1]):
            col_names.append(f"lead_{lead}_sample_{sample}")
    col_names.append("target")
    
    # Create DataFrame
    df = pd.DataFrame(X_flat, columns=col_names[:-1])
    df["target"] = y
    
    # Save subset for CSV (full dataset might be too large)
    if len(df) > 10000:
        df_sample = df.sample(n=10000, random_state=42)
        df_sample.to_csv(output_path / "mitdb_ecg_sample.csv", index=False)
        print("[SUCCESS] Saved: mitdb_ecg_sample.csv (10k samples)")
    else:
        df.to_csv(output_path / "mitdb_ecg_data.csv", index=False)
        print("[SUCCESS] Saved: mitdb_ecg_data.csv")
    
    # Save metadata
    metadata = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "source": "https://physionet.org/content/mitdb/1.0.0/",
        "total_segments": int(X.shape[0]),
        "segment_length": int(segment_length),
        "num_leads": int(X.shape[2]),
        "sampling_rate": "360 Hz",
        "label_mapping": {i: name for i, name in enumerate(label_names)},
        "class_distribution": {name: int(np.sum(y == i)) for i, name in enumerate(label_names)},
        "records_processed": record_names
    }
    
    import json
    with open(output_path / "mitdb_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("[SUCCESS] Saved: mitdb_metadata.json")
    
    print(f"\n[SUCCESS] MIT-BIH dataset processing completed!")
    print(f"Files available for ML pipeline:")
    print(f"  - mitdb_ecg_data.npz - Full dataset (recommended)")
    print(f"  - X.npy + y.npy - Separate arrays")
    print(f"  - mitdb_ecg_*.csv - CSV format")
    
    return X_normalized, y, record_info

def visualize_sample_data(data_path="data", n_samples=3):
    """Visualize sample ECG data"""
    try:
        # Load data
        data = np.load(f"{data_path}/mitdb_ecg_data.npz")
        X, y = data['X'], data['y']
        label_names = data['label_names']
        
        print(f"\nVisualizing {n_samples} sample ECG segments...")
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 8))
        if n_samples == 1:
            axes = [axes]
        
        for i in range(min(n_samples, len(X))):
            # Plot both leads
            axes[i].plot(X[i, :, 0], label=f'Lead I', alpha=0.8)
            if X.shape[2] > 1:
                axes[i].plot(X[i, :, 1], label=f'Lead II', alpha=0.8)
            
            axes[i].set_title(f'Sample {i+1}: {label_names[y[i]]} (Class {y[i]})')
            axes[i].set_xlabel('Sample')
            axes[i].set_ylabel('Normalized Amplitude')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{data_path}/sample_ecg_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[SUCCESS] Visualization saved to: {data_path}/sample_ecg_visualization.png")
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")

def main():
    """Main function to download and process MIT-BIH data"""
    print("MIT-BIH Arrhythmia Database Processor")
    print("=" * 80)
    
    # Step 1: Download data
    downloaded_records, mitdb_path = download_mitdb_data(
        data_dir="data",
        records_to_download=None  # None = download subset for demo
    )
    
    if not downloaded_records:
        print("❌ No records were downloaded. Exiting.")
        return
    
    # Step 2: Process records
    X, y, record_info = process_mitdb_dataset(
        mitdb_path=mitdb_path,
        output_dir="data",
        segment_length=1000  # ~2.8 seconds at 360 Hz
    )
    
    if X is not None:
        # Step 3: Create visualization
        visualize_sample_data("data", n_samples=3)
        
        print(f"\n[READY] Ready to run ML pipeline!")
        print(f"Run: python main.py")
        print(f"The pipeline will automatically detect and process the MIT-BIH ECG data.")
    else:
        print("[ERROR] Data processing failed.")

if __name__ == "__main__":
    main()