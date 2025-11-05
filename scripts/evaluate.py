import argparse
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import json
from pathlib import Path

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate model performance")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output_metrics", required=False)
    return p

def main():
    args = build_parser().parse_args()

    # Load the trained model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Load dataset
    df = pd.read_csv(args.data)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Make predictions
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc:.4f}")

    # Save metrics (ensure directory exists)
    if args.output_metrics:
        Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
        metrics = {"accuracy": acc}
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f)
        print(f"Metrics saved to {args.output_metrics}")

if __name__ == "__main__":
    main()
