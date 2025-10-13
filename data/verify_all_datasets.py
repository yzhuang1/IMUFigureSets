#!/usr/bin/env python
"""
Verification script to check all datasets can be loaded correctly
"""

import numpy as np
import json
from pathlib import Path

def verify_dataset(dataset_path):
    """Verify a single dataset has all required files and can be loaded"""
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name

    print(f"\n{'='*60}")
    print(f"Verifying {dataset_name}")
    print(f"{'='*60}")

    # Check for required files
    x_file = dataset_path / "X.npy"
    y_file = dataset_path / "y.npy"

    # Find metadata file
    metadata_files = list(dataset_path.glob("*metadata.json"))

    if not x_file.exists():
        print(f"❌ Missing X.npy")
        return False

    if not y_file.exists():
        print(f"❌ Missing y.npy")
        return False

    if not metadata_files:
        print(f"❌ Missing metadata.json")
        return False

    metadata_file = metadata_files[0]

    print(f"✓ Found all required files")

    # Load and verify data
    try:
        X = np.load(x_file)
        y = np.load(y_file)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"\nData shapes:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")

        print(f"\nMetadata:")
        dataset_name = metadata.get('dataset_name') or metadata.get('dataset', 'N/A')
        print(f"  Dataset: {dataset_name}")

        n_samples = metadata.get('n_samples') or metadata.get('total_segments', 'N/A')
        if isinstance(n_samples, int):
            print(f"  Samples: {n_samples:,}")
        else:
            print(f"  Samples: {n_samples}")

        n_features = metadata.get('n_features', 'N/A')
        print(f"  Features: {n_features}")

        n_classes = metadata.get('n_classes', 'N/A')
        print(f"  Classes: {n_classes}")

        # Handle class names from different metadata formats
        class_names = metadata.get('class_names')
        if not class_names:
            # Try label_mapping for dataset1 format
            label_mapping = metadata.get('label_mapping')
            if label_mapping:
                class_names = list(label_mapping.values())

        if class_names:
            print(f"  Class names: {class_names}")

        # Handle class distribution
        class_dist = metadata.get('class_distribution')
        if class_dist:
            print(f"\n  Class distribution:")
            for class_name, count in class_dist.items():
                if isinstance(count, int):
                    print(f"    {class_name}: {count:,}")
                else:
                    print(f"    {class_name}: {count}")

        # Verify consistency
        if len(X) != len(y):
            print(f"\n❌ Inconsistent lengths: X has {len(X)} samples, y has {len(y)} samples")
            return False

        expected_samples = metadata.get('n_samples') or metadata.get('total_segments', -1)
        if len(X) != expected_samples and expected_samples != -1:
            print(f"\n⚠️  Warning: Metadata says {expected_samples} samples, but X has {len(X)} samples")

        print(f"\n✓ Dataset verified successfully")
        return True

    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return False

def main():
    """Verify all datasets"""
    base_path = Path(__file__).parent

    results = {}

    # Verify datasets 1-10
    for i in range(1, 11):
        dataset_path = base_path / f"dataset{i}"
        if dataset_path.exists():
            results[f"dataset{i}"] = verify_dataset(dataset_path)
        else:
            print(f"\n⚠️  Dataset{i} directory not found")
            results[f"dataset{i}"] = False

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for dataset, status in results.items():
        symbol = "✓" if status else "❌"
        print(f"{symbol} {dataset}")

    print(f"\n{passed}/{total} datasets verified successfully")

    if passed == total:
        print("\n🎉 All datasets are ready for the pipeline!")
    else:
        print("\n⚠️  Some datasets need attention")

if __name__ == "__main__":
    main()
