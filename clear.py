#!/usr/bin/env python3
"""
Clear script to clean up generated files while preserving documentation.

This script deletes all content in the following folders except for README.md files:
- logs/
- literature_reviews/
- generated_training_functions/
"""

import os
import shutil
import sys
from pathlib import Path


def clear_folder_contents(folder_path, preserve_files=None):
    """
    Clear all contents of a folder except for specified files to preserve.

    Args:
        folder_path (str): Path to the folder to clear
        preserve_files (list): List of filenames to preserve (default: ['README.md'])
    """
    if preserve_files is None:
        preserve_files = ['README.md']

    folder = Path(folder_path)

    if not folder.exists():
        print(f"Folder {folder_path} does not exist, skipping...")
        return

    if not folder.is_dir():
        print(f"{folder_path} is not a directory, skipping...")
        return

    deleted_count = 0
    preserved_count = 0

    print(f"Clearing contents of {folder_path}...")

    for item in folder.iterdir():
        if item.name in preserve_files:
            print(f"  Preserving: {item.name}")
            preserved_count += 1
            continue

        try:
            if item.is_file():
                item.unlink()
                print(f"  Deleted file: {item.name}")
                deleted_count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"  Deleted directory: {item.name}")
                deleted_count += 1
        except Exception as e:
            print(f"  Error deleting {item.name}: {e}")

    print(f"  Summary: {deleted_count} items deleted, {preserved_count} items preserved\n")


def main():
    """Main function to clear specified folders."""
    print("ML Pipeline Cleanup Script")
    print("=" * 40)
    print("This script will delete all content in the following folders:")
    print("- logs/")
    print("- literature_reviews/")
    print("- generated_training_functions/")
    print("README.md files will be preserved in each folder.")
    print()

    # Ask for confirmation
    response = input("Do you want to continue? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        sys.exit(0)

    print("\nStarting cleanup...")
    print("-" * 40)

    # Folders to clear
    folders_to_clear = [
        "logs",
        "literature_reviews",
        "generated_training_functions",
    ]

    # Clear each folder
    for folder in folders_to_clear:
        clear_folder_contents(folder)

    print("Cleanup completed!")


if __name__ == "__main__":
    main()