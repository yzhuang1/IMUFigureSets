# Datasets Summary

All datasets (1-10) have been standardized to the format required by the pipeline:
- **X.npy**: NumPy array containing features
- **y.npy**: NumPy array containing labels
- **metadata.json**: Dataset description and statistics

## Dataset Overview

### Dataset 1: MIT-BIH Arrhythmia Database
- **Samples**: 62,352
- **Features**: ECG signals (2 leads, 1000 samples)
- **Classes**: 5 (Normal, Bundle Branch Block, Atrial Premature, Ventricular, Other)
- **Files**: X.npy, y.npy, mitdb_metadata.json

### Dataset 2: Human Activity Recognition Using Smartphones
- **Samples**: 10,299
- **Features**: 561 (time and frequency domain features from accelerometer and gyroscope)
- **Classes**: 6 (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying)
- **Files**: X.npy, y.npy, har_metadata.json

### Dataset 3: ISRUC-SLEEP Subgroup 1
- **Samples**: 89,283
- **Features**: 6 EEG channels × 6000 samples (30-second epochs)
- **Classes**: 5 (Wake, N1, N2, N3, REM)
- **Files**: X.npy, y.npy, sleep_metadata.json

### Dataset 4: WESAD - Wearable Stress and Affect Detection
- **Samples**: 31,470,603
- **Features**: 8 (ACC X/Y/Z, ECG, EMG, EDA, Temperature, Respiration)
- **Classes**: 4 (Baseline, Stress, Amusement, Meditation)
- **Files**: X.npy, y.npy, wesad_metadata.json
- **Processing Script**: [process_wesad.py](dataset4/process_wesad.py)

### Dataset 5: Combined Physiological Data
- **Samples**: 91,340
- **Features**: 4 (HR, EDA, ACC, BVP)
- **Classes**: 1 (continuous data - may need relabeling)
- **Files**: X.npy, y.npy, dataset5_metadata.json
- **Processing Script**: [process_dataset5.py](dataset5/process_dataset5.py)
- **Note**: This dataset appears to be continuous data without clear labels. May need manual labeling or use as regression task.

### Dataset 6: Wearable Sensor Data - Activity and Injury Risk
- **Samples**: 500
- **Features**: 4 (Heart Rate, Respiration Rate, Body Temperature, Motion Activity Level)
- **Classes**: 3 (Risk Level 0, 1, 2)
- **Files**: X.npy, y.npy, dataset6_metadata.json
- **Processing Script**: [process_dataset6.py](dataset6/process_dataset6.py)

### Dataset 7: Merged Sensor Data - IMU and Physiological
- **Samples**: 11,509,051
- **Features**: 6 (ACC X/Y/Z, EDA, HR, Temperature)
- **Classes**: 3 (Activity 0, 1, 2)
- **Files**: X.npy, y.npy, dataset7_metadata.json
- **Processing Script**: [process_dataset7.py](dataset7/process_dataset7.py)

### Dataset 8: Playground Series S3E24 - Smoker Classification
- **Samples**: 159,256
- **Features**: 22 (age, height, weight, blood pressure, cholesterol, etc.)
- **Classes**: 2 (Non-Smoker, Smoker)
- **Files**: X.npy, y.npy, dataset8_metadata.json
- **Processing Script**: [process_dataset8.py](dataset8/process_dataset8.py)

### Dataset 9: Mental Health Dataset
- **Samples**: 1,547
- **Features**: 13 (audio features, IMU sensors, environmental factors)
- **Classes**: 3 (Anxious, Depressed, Stressed)
- **Files**: X.npy, y.npy, dataset9_metadata.json
- **Processing Script**: [process_dataset9.py](dataset9/process_dataset9.py)

### Dataset 10: Biosensor Product Design Data
- **Samples**: 5
- **Features**: 11 (Age, Cultural Element, EEG features, PAD model)
- **Classes**: 2 (High Satisfaction, Medium Satisfaction)
- **Files**: X.npy, y.npy, dataset10_metadata.json
- **Processing Script**: [process_dataset10.py](dataset10/process_dataset10.py)
- **Note**: Very small dataset - may not be suitable for deep learning without data augmentation.

## Usage

All datasets can now be loaded by the pipeline using:

```python
from adapters.universal_converter import UniversalConverter

converter = UniversalConverter()

# Load dataset
data_path = "data/dataset4"  # or any other dataset
dataset, data_profile = converter.convert_to_torch_dataset(data_path)
```

The pipeline will automatically detect and load the X.npy, y.npy, and metadata.json files.

## Notes

1. **Dataset 5** may need additional processing or labeling as it appears to be continuous physiological data.
2. **Dataset 10** has only 5 samples, which is too small for meaningful machine learning. Consider combining with more data or using transfer learning.
3. All processing scripts are idempotent and can be re-run to regenerate the standardized files.
4. Large datasets (4, 7) may require significant memory during processing and training.

## File Sizes

- Dataset 4: 1.9GB (X.npy), 121MB (y.npy)
- Dataset 7: 527MB (X.npy), 88MB (y.npy)
- Dataset 8: 27MB (X.npy), 1.3MB (y.npy)
- Other datasets: < 10MB each
