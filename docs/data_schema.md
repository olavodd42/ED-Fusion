# Data Schema

This document describes the data structures and formats used in ED-Fusion.

## Dataset Structure

### Input Data Format

The input data should follow this structure:

```
data/
├── raw/
│   ├── train/
│   ├── val/
│   └── test/
└── processed/
    ├── train/
    ├── val/
    └── test/
```

## Data Fields

### Required Fields

- **id**: Unique identifier for each sample
- **features**: Feature vector or tensor
- **label**: Ground truth label (for supervised learning)

### Optional Fields

- **metadata**: Additional information about the sample
- **timestamp**: Time information if applicable

## Data Types

- Features: numpy array or torch tensor
- Labels: integer or float values
- Metadata: dictionary with flexible schema

## Preprocessing Steps

1. Data cleaning and normalization
2. Feature extraction
3. Data augmentation (if applicable)
4. Train/validation/test split

## Data Loading

Use the data loaders provided in the `data/` module to load and process data efficiently.
