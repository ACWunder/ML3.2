# src/models/evaluate_deep.py

import sys
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.models.data import get_dataloaders

def evaluate_deep(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(cfg)

    # Modell aufsetzen wie beim Training
    if cfg["model"]["name"] == "distilbert":
        from transformers import DistilBertForSequenceClassification
        model = DistilBertForSequenceClassification.from_pretrained(
            cfg["model"]["distilbert"]["pretrained_model_name"],
            num_labels=2
        )
    else:
        from src.models.bilstm import BiLSTMClassifier
        model = BiLSTMClassifier(cfg)

    ckpt = cfg["training"]["best_ckpt_path"]
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    # Inferenz
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                labels = batch["labels"]
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                logits = out.logits if hasattr(out, "logits") else out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # Metriken
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Report-Pfad aus config
    reports_cfg = cfg["reports"]["deep_distilbert"]
    df = pd.DataFrame({
        "accuracy":  [acc],
        "precision": [p],
        "recall":    [r],
        "f1":        [f]
    })
    df.to_csv(reports_cfg["metrics_csv"], index=False)

    # Konfusionsmatrix plotten
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.xticks([0,1], reports_cfg["labels"], rotation=45)
    plt.yticks([0,1], reports_cfg["labels"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(reports_cfg["confusion_matrix_png"])

    print(f"Deep eval → acc={acc:.4f}, prec={p:.4f}, rec={r:.4f}, f1={f:.4f}")

if __name__ == "__main__":
    cfg = load_config(sys.argv[1])
    evaluate_deep(cfg)
