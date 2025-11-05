# scripts/train.py
import argparse
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from pathlib import Path

def build_parser():
    p = argparse.ArgumentParser(description="Train model on Titanic features")
    p.add_argument("--input", required=True)
    p.add_argument("--output_model", required=True)
    return p

def main():
    args = build_parser().parse_args()
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    with open(args.output_model, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {args.output_model}")

if __name__ == "__main__":
    main()
