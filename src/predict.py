# src/predict.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


LABELS = {0: "여유", 1: "보통", 2: "혼잡"}


def predict_one(model_path: Path, line: str, station: str, hour: int, dow: int) -> dict:
    model = joblib.load(model_path)
    X = pd.DataFrame([{
        "line": line,
        "station": station,
        "hour": hour,
        "dow": dow
    }])
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0].tolist()
    return {
        "input": {"line": line, "station": station, "hour": hour, "dow": dow},
        "prediction": {"label": pred, "name": LABELS.get(pred, "unknown")},
        "probabilities": proba
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="models/model.joblib")
    p.add_argument("--line", type=str, required=True)
    p.add_argument("--station", type=str, required=True)
    p.add_argument("--hour", type=int, required=True)
    p.add_argument("--dow", type=int, required=True, help="0=월 ... 6=일")
    args = p.parse_args()

    res = predict_one(Path(args.model), args.line, args.station, args.hour, args.dow)
    print(res)


if __name__ == "__main__":
    main()
