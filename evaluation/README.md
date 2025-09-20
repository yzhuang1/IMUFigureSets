# Evaluation

This folder contains the main pipeline orchestrator for the AI-enhanced machine learning pipeline.

## Key Components

- **`code_generation_pipeline_orchestrator.py`** - Main pipeline orchestrator that manages ML pipeline with AI-generated training functions
- **`CodeGenerationPipelineOrchestrator`** - Orchestrates the complete workflow from data processing to model evaluation

## Functionality

- Coordinates the entire ML pipeline execution
- Manages AI-generated training function execution
- Uses training function validation metrics (eliminates preprocessing mismatches)
- Handles single-pass, fail-fast pipeline execution
- Integrates Bayesian optimization with AI-generated model architectures

This module ties together all components of the AI-enhanced machine learning pipeline. Each AI-generated training function handles its own evaluation with proper preprocessing, ensuring consistency between training and validation phases.