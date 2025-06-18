# src/models/data.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from transformers import DistilBertTokenizerFast

class TextDataset(Dataset):
    def __init__(self, texts, labels, cfg, split: str):
        self.texts  = texts
        self.labels = labels
        self.cfg    = cfg
        self.model_name = cfg["model"]["name"]

        if self.model_name == "distilbert":
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                cfg["model"]["distilbert"]["pretrained_model_name"]
            )
            self.max_len = cfg["model"]["distilbert"].get("max_length", 128)
        else:
            if split == "train":
                self.build_vocab(texts)
            self.vocab   = cfg["model"]["bilstm"]["vocab"]
            self.unk_idx = self.vocab["<UNK>"]
            self.pad_idx = self.vocab["<PAD>"]
            self.max_len = cfg["model"]["bilstm"].get("max_length", 100)

    def build_vocab(self, texts):
        freq = {}
        for txt in texts:
            for tok in txt.split():
                freq[tok] = freq.get(tok, 0) + 1
        min_freq = self.cfg["model"]["bilstm"].get("min_freq", 2)
        tokens  = [t for t,n in freq.items() if n >= min_freq]
        self.vocab = {t: i+2 for i,t in enumerate(tokens)}
        self.vocab["<UNK>"] = 0
        self.vocab["<PAD>"] = 1
        self.cfg["model"]["bilstm"]["vocab"] = self.vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt   = self.texts[idx]
        label = int(self.labels[idx])

        if self.model_name == "distilbert":
            enc = self.tokenizer(
                txt,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(label, dtype=torch.long),
            }
        else:
            idxs = [ self.vocab.get(w, self.unk_idx) for w in txt.split() ]
            if len(idxs) < self.max_len:
                idxs += [self.pad_idx] * (self.max_len - len(idxs))
            else:
                idxs = idxs[: self.max_len]
            return {
                "input_ids": torch.tensor(idxs, dtype=torch.long),
                "labels":     torch.tensor(label, dtype=torch.long),
            }


def get_dataloaders(cfg):
    df_tr = pd.read_csv(cfg["data"]["train_csv"])
    df_va = pd.read_csv(cfg["data"]["val_csv"])
    df_te = pd.read_csv(cfg["data"]["test_csv"])

    train_ds = TextDataset(df_tr["full_text"].tolist(),
                           df_tr["label"].tolist(), cfg, split="train")
    val_ds   = TextDataset(df_va["full_text"].tolist(),
                           df_va["label"].tolist(), cfg, split="val")
    test_ds  = TextDataset(df_te["full_text"].tolist(),
                           df_te["label"].tolist(), cfg, split="test")

    batch_size = cfg["training"]["batch_size"]

    def collate_fn(batch):
        if cfg["model"]["name"] == "distilbert":
            input_ids = torch.stack([x["input_ids"]      for x in batch])
            attn_mask = torch.stack([x["attention_mask"] for x in batch])
            labels    = torch.stack([x["labels"]         for x in batch])
            return {"input_ids": input_ids,
                    "attention_mask": attn_mask,
                    "labels": labels}
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            labels    = torch.stack([x["labels"]    for x in batch])
            return input_ids, labels

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False,
                              collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
