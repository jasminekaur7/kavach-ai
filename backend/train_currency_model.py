"""Train a real currency-authenticity classifier on the UCI "Bank Note
Authentication" dataset (1372 rows, 4 wavelet-transform features extracted
from banknote images: variance, skewness, curtosis, entropy).

This gives you an honest, real accuracy number for the pitch deck, separate
from the live-camera heuristic in agents.py (which works on raw uploaded
photos rather than pre-extracted wavelet features).

Run:
    pip install ucimlrepo scikit-learn joblib
    python train_currency_model.py

If ucimlrepo can't reach the network from your machine, download the CSV
manually from either of:
  - https://archive.ics.uci.edu/dataset/267/banknote+authentication
  - https://www.kaggle.com/datasets/vivekgediya/banknote-authentication-uci-data
and place it as data/banknote.csv with columns:
  variance, skewness, curtosis, entropy, class
"""
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

COLUMNS = ["variance", "skewness", "curtosis", "entropy", "class"]
LOCAL_CSV = Path(__file__).parent / "data" / "banknote.csv"


def load_data():
    if LOCAL_CSV.exists():
        import pandas as pd
        df = pd.read_csv(LOCAL_CSV, header=None, names=COLUMNS)
        return df.drop(columns="class"), df["class"]

    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=267)  # Bank Note Authentication
        return dataset.data.features, dataset.data.targets.iloc[:, 0]
    except Exception as e:
        sys.exit(
            "Could not load the dataset automatically "
            f"({e}).\nDownload it manually — see the instructions at the "
            "top of this file — and save it to data/banknote.csv."
        )


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["genuine", "counterfeit"]))

    out = Path(__file__).parent / "currency_model.pkl"
    joblib.dump(model, out)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
