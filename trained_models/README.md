# Trained Models

This folder contains saved model weights and checkpoints from successful training runs.

## Contents

- **Model Checkpoints** - Saved PyTorch model states (.pth files)
- **Best Models** - Top-performing models from Bayesian optimization
- **Model Metadata** - Configuration and performance information
- **Serialized Models** - Complete model objects for deployment

## Structure

Models are typically saved with descriptive names including:
- Architecture type
- Dataset identifier
- Performance metrics
- Timestamp

## Purpose

- Preserves trained models for reuse and deployment
- Stores best-performing configurations from BO optimization
- Enables model comparison and analysis
- Supports model versioning and rollback

Models are automatically saved during training by the training function executor when significant improvements are achieved.