import subprocess
import os
from pathlib import Path
from tqdm import tqdm

# Set the directory paths
rar_dir = Path("/home/shiyuanduan/Documents/Projects/GPTLAB/IMUFigureSets/data/dataset3/ISRUC_Subgroup1_raw")
extract_dir = Path("/home/shiyuanduan/Documents/Projects/GPTLAB/IMUFigureSets/data/dataset3/ISRUC_Subgroup1_extracted")
unrar_path = "/home/shiyuanduan/.local/bin/unrar"

# Create extraction directory
extract_dir.mkdir(exist_ok=True)

# Get all RAR files
rar_files = sorted(rar_dir.glob("*.rar"))

print(f"Found {len(rar_files)} RAR files to extract")

# Extract each RAR file
for rar_file in tqdm(rar_files):
    try:
        # Extract to a subdirectory named after the RAR file (without extension)
        subject_dir = extract_dir / rar_file.stem
        subject_dir.mkdir(exist_ok=True)

        # Run unrar with -o+ to overwrite without prompting
        result = subprocess.run(
            [unrar_path, "x", "-o+", str(rar_file), str(subject_dir) + "/"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✓ Extracted {rar_file.name}")
        else:
            print(f"✗ Error extracting {rar_file.name}: {result.stderr}")
    except Exception as e:
        print(f"✗ Error extracting {rar_file.name}: {e}")

print(f"\nExtraction complete! Files extracted to: {extract_dir}")
