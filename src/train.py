# src/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier




def train(
    processed_csv: Path,
    model_out: Path,
    metrics_out: Path,
    random_state: int = 42
) -> dict:
    df = pd.read_csv(processed_csv)

    # 특징/타깃
    X = df[["line", "station", "hour", "dow"]].copy()
    y = df["congestion"].astype(int)

    # 학습/평가 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    cat_features = ["line", "station"]
    num_features = ["hour", "dow"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ],
        remainder="drop",
    )

    # 모델(초기 기준으로 RandomForest 추천)
    clf = SGDClassifier(
        loss="log_loss",  # 다중 클래스 로지스틱
        max_iter=15,  # 매우 빠름
        tol=1e-3,
        class_weight="balanced",
        random_state=42
    )

    pipe = Pipeline([("preprocess", pre), ("model", clf)])
    pipe.fit(X_train, y_train)

    # 평가
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "test_size": 0.2,
        "random_state": random_state,
        "classes": {0: "여유", 1: "보통", 2: "혼잡"},
        "classification_report": report,
        "confusion_matrix": cm,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # 저장
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed.csv")
    p.add_argument("--model_out", type=str, default="models/model.joblib")
    p.add_argument("--metrics_out", type=str, default="outputs/metrics.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    metrics = train(Path(args.data), Path(args.model_out), Path(args.metrics_out), random_state=args.seed)
    print("[OK] training done")
    # 콘솔에도 핵심만 출력
    acc = metrics["classification_report"]["accuracy"]
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
