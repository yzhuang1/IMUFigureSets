#!/usr/bin/env python3
"""
Master script to download all datasets (4-10)
Requires Kaggle API credentials to be configured
"""

import os
import subprocess
import sys

def check_kaggle_auth():
    """Check if Kaggle credentials are configured"""
    kaggle_json = os.path.expanduser("~/.config/kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle API credentials not found!")
        print("\nTo set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token' to download kaggle.json")
        print("4. Move it to: ~/.config/kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.config/kaggle/kaggle.json")
        return False
    return True

def download_dataset(dataset_num):
    """Download a specific dataset"""
    dataset_dir = f"dataset{dataset_num}"
    script_path = os.path.join(dataset_dir, "download.py")

    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Downloading Dataset {dataset_num}")
    print(f"{'='*60}")

    try:
        # Use sys.executable to ensure we use the same Python interpreter
        result = subprocess.run(
            [sys.executable, "download.py"],
            cwd=dataset_dir,
            check=True,
            capture_output=False
        )
        print(f"✅ Dataset {dataset_num} downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset {dataset_num}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error for dataset {dataset_num}: {e}")
        return False

def main():
    """Download all datasets"""
    print("Dataset Download Manager")
    print("="*60)

    # Check Kaggle authentication
    if not check_kaggle_auth():
        sys.exit(1)

    # Download datasets 4-10
    datasets = list(range(4, 11))
    success_count = 0
    failed_datasets = []

    for dataset_num in datasets:
        if download_dataset(dataset_num):
            success_count += 1
        else:
            failed_datasets.append(dataset_num)

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully downloaded: {success_count}/{len(datasets)}")

    if failed_datasets:
        print(f"❌ Failed datasets: {', '.join(map(str, failed_datasets))}")
    else:
        print("🎉 All datasets downloaded successfully!")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
