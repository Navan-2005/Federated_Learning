"""Evaluate a saved global model on a pixel CSV (pixel_0..pixel_783 + label).

Usage:
    python evaluate_model.py --data C:\\path\\to\\test.csv [--model global_model_async.pt]

Use data the model did not train on (e.g. a held-out test file) for an honest score.
"""
import argparse
import os

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from fixed_preprocessing import NUM_CLASSES, NUM_FEATURES, TARGET, preprocess, validate
from network import ANN_Model

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV with pixel_0..pixel_783 and label")
    p.add_argument("--model", default=os.path.join(HERE, "global_model_async.pt"))
    args = p.parse_args()

    df = validate(pd.read_csv(args.data))
    X, y = preprocess(df)

    model = ANN_Model(input_features=NUM_FEATURES, out_features=NUM_CLASSES)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X)).argmax(dim=1).numpy()

    print(f"Rows evaluated: {len(y)}")
    print(f"Accuracy: {(preds == y).mean():.4f}\n")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y, preds, labels=list(range(NUM_CLASSES))))
    print("\nPer-class report:")
    print(classification_report(y, preds, labels=list(range(NUM_CLASSES)), zero_division=0))


if __name__ == "__main__":
    main()
