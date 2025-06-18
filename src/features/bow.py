# src/features/bow.py

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

# src/features/bow.py

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_vectorizer(feat_cfg):
    """
    Liefert einen unfitten Count- oder TfidfVectorizer zurück.
    feat_cfg kommt aus configs/train_shallow.yaml['features'] und enthält:
      - 'ngram_range': [1,2]
      - 'use_tfidf': True/False
      - optional: 'max_df', 'min_df'
    """
    # 1) ngram_range als tuple
    ngram_range = tuple(feat_cfg['ngram_range'])

    # 2) gemeinsame Parameter
    kwargs = {
        'ngram_range': ngram_range,
        'max_df':      feat_cfg.get('max_df', 1.0),
        'min_df':      feat_cfg.get('min_df', 1),
    }

    # 3) Count- vs. TF–IDF-Vektorizer
    if feat_cfg.get('use_tfidf', False):
        return TfidfVectorizer(**kwargs)
    else:
        return CountVectorizer(**kwargs)


def fit_and_save_vectorizer(cfg):
    """
    Nutze das train.csv, um den Vectorizer zu fitten und auf
    disk unter cfg['features']['vectorizer_path'] zu speichern.
    """
    df = pd.read_csv(cfg['split']['train_csv'])
    vec = build_vectorizer(cfg['features'])
    vec.fit(df['full_text'])
    joblib.dump(vec, cfg['features']['vectorizer_path'])
