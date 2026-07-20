"""Train a real voice-spoof classifier on downloaded labeled audio clips,
replacing the guessed-threshold heuristic in agents.py's check_voice.

Download real + fake clips first, e.g. the Fake-or-Real (FoR) dataset's
"for-2sec" folders (short clips, easy to test quickly):
    https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

Then run (start with ~30-50 clips per class — that's enough for logistic
regression on these 18 hand-crafted features; more only helps if you have
it easily available):
    pip install librosa soundfile scikit-learn joblib
    python train_voice_model.py path/to/real_folder path/to/fake_folder

This writes voice_model.pkl next to this script. check_voice() in agents.py
automatically picks it up on the next server restart — no code change needed.
"""
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agents import extract_voice_features  # same features used at inference time


def load_folder(folder: Path, label: int) -> tuple:
    X, y = [], []
    paths = sorted(folder.glob("*.wav")) + sorted(folder.glob("*.flac"))
    for path in paths:
        try:
            audio, sr = librosa.load(path, sr=16000, mono=True)
            features, _, _ = extract_voice_features(audio, sr)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
    return X, y


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python train_voice_model.py <real_folder> <fake_folder>")
    real_dir, fake_dir = Path(sys.argv[1]), Path(sys.argv[2])

    print(f"loading real clips from {real_dir} ...")
    X_real, y_real = load_folder(real_dir, 0)
    print(f"loading fake clips from {fake_dir} ...")
    X_fake, y_fake = load_folder(fake_dir, 1)
    print(f"loaded {len(X_real)} real, {len(X_fake)} fake clips")

    if len(X_real) < 10 or len(X_fake) < 10:
        sys.exit("need at least ~10 clips per class for a meaningful test split — download more first")

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_train), y_train)

    preds = model.predict(scaler.transform(X_test))
    print(f"\nheld-out accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, target_names=["genuine", "spoofed"]))

    out = Path(__file__).parent / "voice_model.pkl"
    joblib.dump({"scaler": scaler, "model": model}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()