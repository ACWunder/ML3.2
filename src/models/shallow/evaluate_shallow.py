# src/models/evaluate_shallow.py

import sys
import time
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from src.utils.config import load_config


def evaluate(cfg):
    # 1) Daten laden
    test_df = pd.read_csv(cfg['split']['test_csv'])
    X_test = test_df['full_text']
    y_test = test_df['label']

    # 2) Modell laden
    model = joblib.load(cfg['models']['shallow']['best_model_path'])

    # 3) Inferenz + Laufzeit messen
    start = time.time()
    preds = model.predict(X_test)
    infer_time = time.time() - start

    # 4) Metriken berechnen
    acc = accuracy_score(y_test, preds)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_test, preds)

    # 5) Report als CSV speichern
    report = {
        'accuracy': [acc],
        'precision': [p],
        'recall': [r],
        'f1_score': [f],
        'inference_time_sec': [infer_time],
    }
    report_df = pd.DataFrame(report)
    report_df.to_csv(cfg['reports']['shallow']['metrics_csv'], index=False)

    # 6) Confusion-Matrix plotten und speichern
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(cfg['reports']['shallow']['labels']))
    plt.xticks(tick_marks, cfg['reports']['shallow']['labels'], rotation=45)
    plt.yticks(tick_marks, cfg['reports']['shallow']['labels'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cfg['reports']['shallow']['confusion_matrix_png'])

    print(f"✔ Accuracy:  {acc:.4f}")
    print(f"✔ Precision: {p:.4f}")
    print(f"✔ Recall:    {r:.4f}")
    print(f"✔ F1-score:  {f:.4f}")
    print(f"✔ Inference time: {infer_time:.3f}s")
    print(f"✔ Metrics saved to {cfg['reports']['shallow']['metrics_csv']}")
    print(f"✔ CM plot saved to {cfg['reports']['shallow']['confusion_matrix_png']}")


if __name__ == "__main__":
    # Aufruf: python -m src.models.evaluate_shallow configs/preprocess.yaml
    cfg = load_config(sys.argv[1])
    evaluate(cfg)
