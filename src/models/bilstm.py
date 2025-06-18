# src/models/bilstm.py

import torch
import torch.nn as nn
from types import SimpleNamespace

class BiLSTMClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vocab_size = len(cfg["model"]["bilstm"]["vocab"])
        embed_dim  = cfg["model"]["bilstm"].get("embed_dim", 100)
        hidden_dim = cfg["model"]["bilstm"].get("hidden_dim", 128)
        num_layers = cfg["model"]["bilstm"].get("num_layers", 1)
        dropout    = cfg["model"]["bilstm"].get("dropout", 0.2)

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=cfg["model"]["bilstm"]["vocab"]["<PAD>"]
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        # Anzahl der Labels holen wir direkt aus cfg["training"]
        num_labels = cfg["training"].get("num_labels", 2)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, labels=None, attention_mask=None):
        # (1) Einbettung + LSTM
        embeds = self.embedding(input_ids)          # (B, L, E)
        outputs, _ = self.lstm(embeds)              # (B, L, 2H)

        # (2) Letzten Zeitschritt extrahieren
        last = outputs[:, -1, :]                    # (B, 2H)
        dropped = self.dropout(last)

        # (3) Klassifikation
        logits = self.fc(dropped)                   # (B, num_labels)

        # (4) Optional Loss berechnen
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # labels muss LongTensor sein, Form: (B,)
            loss = loss_fct(logits, labels)

        # (5) Objekt mit loss & logits zurückgeben
        return SimpleNamespace(loss=loss, logits=logits)
