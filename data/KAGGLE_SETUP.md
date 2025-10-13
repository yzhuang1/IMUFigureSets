# Kaggle API Setup Instructions

## Prerequisites
The Kaggle Python package is already installed.

## Setting Up Kaggle API Credentials

### Step 1: Get Your Kaggle API Token
1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll down to the **API** section
3. Click **Create New Token**
4. This will download a file called `kaggle.json`

### Step 2: Install the Credentials
```bash
# Create the kaggle config directory (if it doesn't exist)
mkdir -p ~/.config/kaggle

# Move the downloaded kaggle.json file
mv ~/Downloads/kaggle.json ~/.config/kaggle/

# Set proper permissions (required for security)
chmod 600 ~/.config/kaggle/kaggle.json
```

### Step 3: Verify Installation
```bash
kaggle datasets list
```

If you see a list of datasets, the setup is successful!

## Downloading All Datasets

Once Kaggle credentials are configured, run:

```bash
cd data/
python3 download_all_datasets.py
```

This will download all 7 datasets (dataset4 through dataset10) from the Excel file.

## Download Individual Datasets

To download a specific dataset:

```bash
cd data/dataset4  # or dataset5, dataset6, etc.
python3 download.py
```

## Dataset Information

| Dataset # | Description | Kaggle Link |
|-----------|-------------|-------------|
| dataset4 | WESAD - Wearable Stress and Affect Detection | [Link](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset) |
| dataset5 | Wearable Sensor Data for Activity Analysis | [Link](https://www.kaggle.com/datasets/oumaymabejaoui/wearable-sensor-data-for-activity-analysis) |
| dataset6 | Wearable Sensor System for Physical Education | [Link](https://www.kaggle.com/datasets/ziya07/wearable-sensor-system-for-physical-education) |
| dataset7 | Nurse Stress Prediction - Wearable Sensors | [Link](https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors) |
| dataset8 | Playground Series S3E24 - Smoker Classification | [Link](https://www.kaggle.com/competitions/playground-series-s3e24/overview) |
| dataset9 | Wearable Sensor Data for Mental Health Prediction | [Link](https://www.kaggle.com/datasets/programmer3/wearable-sensor-data-for-mental-health-prediction) |
| dataset10 | Biosensor-Driven Product Design | [Link](https://www.kaggle.com/datasets/ziya07/biosensor-driven-product-design) |

## Troubleshooting

### Error: "Could not find kaggle.json"
Make sure the file is in `~/.config/kaggle/kaggle.json` and has proper permissions (600).

### Error: "403 Forbidden"
You may need to accept the competition rules on Kaggle's website first (especially for dataset8).

### Error: "404 Not Found"
The dataset may have been removed or renamed. Check the Kaggle link in the table above.
