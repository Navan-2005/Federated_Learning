"""Data-independent preprocessing for the image (pixel CSV) dataset.

Every participant applies the same fixed schema, so no client needs to see
another client's data to agree on feature order and scale. The server and all
clients must use an identical copy of this file.

CSV format: columns pixel_0 ... pixel_783 (a flattened 28x28 image) plus `label`.
"""
import numpy as np

TARGET = "label"
IMAGE_SIDE = 28
NUM_FEATURES = IMAGE_SIDE * IMAGE_SIDE  # 784
NUM_CLASSES = 10
PIXEL_COLUMNS = [f"pixel_{i}" for i in range(NUM_FEATURES)]
REQUIRED_COLUMNS = [*PIXEL_COLUMNS, TARGET]
MIN_ROWS = 50


def validate(df):
    """Return a cleaned copy of df with exactly the required columns, or raise ValueError."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    df = df[REQUIRED_COLUMNS].dropna()
    if len(df) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} valid rows")
    try:
        df = df.astype({TARGET: np.int64})
        df[PIXEL_COLUMNS] = df[PIXEL_COLUMNS].astype(np.float32)
    except (ValueError, TypeError):
        raise ValueError("pixel columns and label must be numeric")
    if not df[TARGET].between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"label must be an integer in 0..{NUM_CLASSES - 1}")
    return df


def preprocess(df):
    """Scale pixels to [0, 1]. Values above 1 are taken to be 0-255 intensities."""
    X = df[PIXEL_COLUMNS].to_numpy(dtype=np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    X = np.clip(X, 0.0, 1.0).astype(np.float32)
    y = df[TARGET].to_numpy().astype(np.int64)
    return X, y
