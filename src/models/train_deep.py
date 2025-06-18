# src/models/train_deep.py

import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import load_config
from src.models.data import get_dataloaders

def train_deep(cfg):
    # 1) Device & DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 2) Modell instanziieren
    if cfg["model"]["name"] == "distilbert":
        from transformers import DistilBertForSequenceClassification
        model = DistilBertForSequenceClassification.from_pretrained(
            cfg["model"]["distilbert"]["pretrained_model_name"],
            num_labels=2
        )
    else:
        from src.models.bilstm import BiLSTMClassifier
        model = BiLSTMClassifier(cfg)

    model.to(device)

    # 3) Optimizer & (nur für BiLSTM) Loss-Funktion
    lr = float(cfg["training"].get("lr", 5e-5))
    optimizer = AdamW(model.parameters(), lr=lr)

    # 4) TensorBoard
    tb_logdir = cfg["training"].get("tb_logdir", "runs/deep_distilbert")
    writer = SummaryWriter(log_dir=tb_logdir)

    best_val_loss = float("inf")
    patience = int(cfg["training"].get("early_stop_patience", 2))
    max_epochs = int(cfg["training"].get("epochs", 3))
    ckpt_path = cfg["training"].get("best_ckpt_path", "models/deep_distilbert/best.pt")

    # 5) Training loop
    patience_counter = 0
    for epoch in range(1, max_epochs + 1):
        # — training
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            # move to device + forward
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs, labels=labels)

            # **immer** outputs.loss nutzen
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)

        # — validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val = total_val_loss / len(val_loader)

        # — Logging & Early Stopping
        writer.add_scalars("Loss", {"train": avg_train, "val": avg_val}, epoch)
        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping.")
                break

    writer.close()
    print("✔ Training complete. Best val loss:", best_val_loss)


if __name__ == "__main__":
    cfg = load_config(sys.argv[1])
    train_deep(cfg)
