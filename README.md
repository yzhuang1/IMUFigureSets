# ML Pipeline Scaffold

This scaffold includes:
- A unified data adapter that converts different dataset types (time series, tabular, image) into PyTorch Datasets.
- A simple model zoo (tiny CNN for time series, MLP for tabular, small CNN for images).
- A training loop and evaluation utilities.
- A stubbed Bayesian Optimization runner interface you can replace with your favorite BO library.

Quick start:
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

